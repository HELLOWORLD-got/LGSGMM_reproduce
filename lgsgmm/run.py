"""Configuration-driven training, resume, smoke validation and two-stage evaluation."""
import argparse
from contextlib import nullcontext
import csv
import json
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .data import MDMTDataset, collate_pairs, to_device
from .loss import matching_loss
from .metrics import association_counts, scores, summarize
from .model import PaperLGSGMM
from .postprocess import refine
from . import distributed

ROOT = Path(__file__).resolve().parents[1]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def make_loader(config, split, limit=None):
    manifest_name = config.get('test_manifest', 'test.jsonl') if split == 'test' else f'{split}.jsonl'
    dataset = MDMTDataset(ROOT / config['data_dir'] / manifest_name,
                          stride=config.get(f'{split}_stride', 1), limit=limit,
                          max_side=config['image_max_side'])
    sampler = None
    if distributed.world_size() > 1:
        sampler = (DistributedSampler(dataset, seed=config['seed']) if split == 'train'
                   else distributed.EvaluationSampler(dataset))
    return DataLoader(dataset, batch_size=config['batch_size'] // distributed.world_size(),
                      shuffle=split == 'train' and sampler is None, sampler=sampler,
                      num_workers=config['workers'], collate_fn=collate_pairs,
                      pin_memory=config['device'].startswith('cuda'))


def synchronize(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize()


def train_epoch(model, loader, optimizer, scaler, config):
    model.train()
    accumulated = 0.
    parts = dict(classification=0., triplet=0.)
    samples = 0
    for batch in tqdm(loader, desc='Train', disable=distributed.rank() != 0):
        optimizer.zero_grad(set_to_none=True)
        for index, sample in enumerate(batch):
            context = (model.no_sync() if isinstance(model, DistributedDataParallel)
                       and index < len(batch) - 1 else nullcontext())
            with context:
                views = to_device(sample, config['device'])
                with torch.autocast(device_type=config['device'].split(':')[0], enabled=scaler.is_enabled()):
                    output = model(views)
                    loss, detail = matching_loss(output, views, **config['loss'])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at {sample["sequence"]}/{sample["frame"]}')
                scaler.scale(loss / len(batch)).backward()
            accumulated += float(loss.detach())
            samples += 1
            for key in parts:
                parts[key] += detail[key]
        scaler.unscale_(optimizer)
        if config['gradient_clip']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
    accumulated, samples, classification, triplet = distributed.reduce_totals(
        [accumulated, samples, parts['classification'], parts['triplet']], config['device'])
    return dict(loss=accumulated / samples, classification=classification / samples,
                triplet=triplet / samples)


@torch.no_grad()
def evaluate(model, loader, config, two_stage=False):
    distributed.broadcast_buffers(model)
    model = distributed.unwrap(model)
    model.eval()
    rows, second_rows = [], []
    loss_total, count = 0., 0
    for batch in tqdm(loader, desc='Evaluate', disable=distributed.rank() != 0):
        for sample in batch:
            views = to_device(sample, config['device'])
            synchronize(config['device'])
            start = time.perf_counter()
            output = model(views)
            synchronize(config['device'])
            network_ms = (time.perf_counter() - start) * 1000
            # Preserve RNG so validation sampling cannot alter the next training epoch.
            loss, _ = matching_loss(output, views, **config['loss'])
            loss_total += float(loss)
            count += 1
            metadata = dict(sequence=sample['sequence'], frame=sample['frame'], network_ms=network_ms)
            rows.append(dict(metadata, **association_counts(views, output['matches'].cpu().numpy())))
            if two_stage:
                post_start = time.perf_counter()
                matches, status = refine(views, output, config['sgmm'])
                second_rows.append(dict(metadata, **association_counts(views, matches),
                                        total_ms=network_ms + (time.perf_counter() - post_start) * 1000,
                                        postprocess=status['status']))
    loss_total, count = distributed.reduce_totals([loss_total, count], config['device'])
    rows = [row for values in distributed.gather(rows) for row in values]
    second_rows = [row for values in distributed.gather(second_rows) for row in values]
    rows.sort(key=lambda row: (row['sequence'], row['frame']))
    second_rows.sort(key=lambda row: (row['sequence'], row['frame']))
    result = dict(loss=loss_total / count, one_stage=summarize(rows), pairs=int(count))
    if two_stage:
        result['two_stage'] = summarize(second_rows)
    return result, rows, second_rows


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda_device=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda_device') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda_device'].cpu())
    elif state.get('cuda') and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'evaluate', 'smoke'])
    parser.add_argument('--config', default='configs/mdmt.json')
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--device')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--val-limit', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--max-side', type=int)
    parser.add_argument('--output-dir')
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--no-pga', action='store_true')
    parser.add_argument('--no-lga', action='store_true')
    parser.add_argument('--one-stage', action='store_true')
    args = parser.parse_args(argv)
    if args.resume and args.checkpoint:
        parser.error('Use only one of --resume and --checkpoint')
    if args.command == 'evaluate' and not args.checkpoint:
        parser.error('evaluate requires --checkpoint')
    for value in (args.limit, args.train_limit, args.val_limit):
        if value is not None and value < 1:
            parser.error('sample limits must be positive')
    config = json.loads((ROOT / args.config).read_text())
    for field in ('device', 'batch_size', 'epochs', 'output_dir'):
        value = getattr(args, field)
        if value is not None:
            config[field] = value
    if args.max_side:
        config['image_max_side'] = args.max_side
    if args.no_pga:
        config['model']['pga'] = False
    if args.no_lga:
        config['model']['lga'] = False
    if args.scratch:
        config['model']['backbone_weights'] = None
    config['device'] = distributed.initialize(config['device'])
    size, rank = distributed.world_size(), distributed.rank()
    seed_all(config['seed'] + rank)
    torch.set_num_threads(min(torch.get_num_threads(), 4 if size > 1 else 8))
    output_dir = ROOT / config['output_dir']
    if args.command == 'smoke' and args.config == 'configs/mdmt.json':
        if not args.output_dir:
            output_dir = ROOT / 'runs/smoke'
        config.update(image_max_side=args.max_side or 160,
                      batch_size=args.batch_size or size, epochs=args.epochs or 1)
    if config['batch_size'] < size or config['batch_size'] % size:
        parser.error('global batch size must be positive and divisible by WORLD_SIZE')
    if rank == 0:
        print(f'DDP world_size={size}, global_batch_size={config["batch_size"]}, '
              f'per_rank_batch_size={config["batch_size"] // size}', flush=True)
    devices = distributed.gather(dict(rank=rank, device=config['device'],
                                     name=torch.cuda.get_device_name() if config['device'].startswith('cuda')
                                     else 'CPU'))
    if rank == 0:
        print(json.dumps(dict(processes=devices)), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.resume or args.checkpoint
    checkpoint = None
    if checkpoint_path:
        with Path(checkpoint_path).open('rb') as stream:
            if stream.read(70).startswith(b'version https://git-lfs'):
                raise ValueError('Checkpoint is a Git LFS pointer. Download its actual content first.')
        # Local training checkpoints include optimizer and Python/NumPy RNG state.
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if args.resume:
            if args.checkpoint:
                parser.error('Use only one of --resume and --checkpoint')
            previous = checkpoint['config']
            for key in ('model', 'loss', 'sgmm', 'image_max_side', 'batch_size', 'train_stride', 'val_stride'):
                if config[key] != previous[key]:
                    raise ValueError(f'Resume configuration mismatch: {key}')
            if checkpoint.get('world_size', 1) != size:
                raise ValueError('Resume requires the checkpoint world size')
        else:
            if args.no_pga or args.no_lga or args.scratch:
                parser.error('Evaluation uses the architecture saved in the checkpoint')
            config['model'] = checkpoint['config']['model']
            for key in ('loss', 'sgmm', 'image_max_side'):
                config[key] = checkpoint['config'][key]
            if args.max_side:
                config['image_max_side'] = args.max_side
        model_config = dict(config['model'], backbone_weights=None)
    else:
        model_config = config['model']
    # Download backbone weights once. Other ranks use the populated shared cache.
    if size > 1 and rank != 0:
        dist.barrier()
    model = PaperLGSGMM(**model_config).to(config['device'])
    if size > 1 and rank == 0:
        dist.barrier()
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    if size > 1 and args.command != 'evaluate':
        model = DistributedDataParallel(
            model, device_ids=[torch.device(config['device']).index]
            if config['device'].startswith('cuda') else None,
            find_unused_parameters=True)
    if rank == 0:
        (output_dir / f'{args.command}_config.json').write_text(json.dumps(config, indent=2) + '\n')
    if args.command == 'evaluate':
        if checkpoint is None:
            parser.error('evaluate requires --checkpoint')
        metrics, rows, second = evaluate(model, make_loader(config, args.split, args.limit), config,
                                         two_stage=not args.one_stage)
        if rank != 0:
            return
        for name, records in [('one_stage', rows), ('two_stage', second)]:
            if records:
                with (output_dir / f'{args.split}_{name}.csv').open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
        (output_dir / f'{args.split}_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        print(json.dumps(metrics, indent=2))
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=config['scheduler_patience'], factor=config['scheduler_factor'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['amp'] and config['device'].startswith('cuda'))
    start_epoch, best = 0, -1.
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        states = checkpoint.get('rng_by_rank')
        restore_rng(states[rank] if states else checkpoint['rng'])
        start_epoch, best = checkpoint['epoch'] + 1, checkpoint['best_mda']
    history_path = output_dir / 'history.jsonl'
    if rank == 0:
        if not args.resume and history_path.exists() and args.command != 'smoke':
            raise FileExistsError(f'Existing training run, use --resume or another --output-dir: {output_dir}')
        if args.command == 'smoke' and history_path.exists():
            history_path.unlink()
    limit = args.limit or (max(2, size * 2) if args.command == 'smoke' else None)
    train_limit = args.train_limit or args.limit or config.get('train_limit') or limit
    val_limit = args.val_limit or args.limit or config.get('val_limit') or limit
    train_loader = make_loader(config, 'train', train_limit)
    val_loader = make_loader(config, 'val', val_limit)
    for epoch in range(start_epoch, config['epochs']):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        training = train_epoch(model, train_loader, optimizer, scaler, config)
        state = rng_state()
        validation, _, _ = evaluate(model, val_loader, config, two_stage=args.command == 'smoke')
        restore_rng(state)
        scheduler.step(validation['loss'])
        mda = validation['one_stage']['sequence_macro']['mda']
        improved = mda > best
        best = max(best, mda)
        states = distributed.gather(rng_state())
        if rank != 0:
            continue
        snapshot = dict(model=distributed.unwrap(model).state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                        epoch=epoch, best_mda=best, config=config, rng=states[0],
                        rng_by_rank=states, world_size=size)
        atomic_save(snapshot, output_dir / 'last.pt')
        if improved:
            atomic_save(snapshot, output_dir / 'best.pt')
        record = dict(epoch=epoch + 1, learning_rate=optimizer.param_groups[0]['lr'],
                      train=training, validation=validation)
        with (output_dir / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record))
    if size > 1:
        dist.barrier()
    if args.command == 'smoke' and rank == 0:
        reloaded = torch.load(output_dir / 'last.pt', map_location='cpu', weights_only=False)
        distributed.unwrap(model).load_state_dict(reloaded['model'], strict=True)
        print('Smoke passed: train, backward, validation, second stage, checkpoint save/reload')


if __name__ == '__main__':
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
