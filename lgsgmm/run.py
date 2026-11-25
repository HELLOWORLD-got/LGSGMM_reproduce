"""Configuration-driven training, resume, smoke validation and two-stage evaluation."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import MDMTDataset, collate_pairs, to_device
from .loss import matching_loss
from .metrics import association_counts, scores, summarize
from .model import PaperLGSGMM
from .postprocess import refine

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
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=split == 'train',
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
    for batch in tqdm(loader, desc='Train'):
        optimizer.zero_grad(set_to_none=True)
        for sample in batch:
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
    return dict(loss=accumulated / samples, **{key: value / samples for key, value in parts.items()})


@torch.no_grad()
def evaluate(model, loader, config, two_stage=False):
    model.eval()
    rows, second_rows = [], []
    loss_total, count = 0., 0
    for batch in tqdm(loader, desc='Evaluate'):
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
    result = dict(loss=loss_total / count, one_stage=summarize(rows), pairs=count)
    if two_stage:
        result['two_stage'] = summarize(second_rows)
    return result, rows, second_rows


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
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
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--max-side', type=int)
    parser.add_argument('--output-dir')
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--no-pga', action='store_true')
    parser.add_argument('--no-lga', action='store_true')
    parser.add_argument('--one-stage', action='store_true')
    args = parser.parse_args(argv)
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
    if config['batch_size'] < 1:
        parser.error('batch size must be positive')
    if config['device'] == 'auto':
        config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed_all(config['seed'])
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    output_dir = ROOT / config['output_dir']
    if args.command == 'smoke' and not args.output_dir:
        output_dir = ROOT / 'runs/smoke'
        config.update(image_max_side=args.max_side or 160, batch_size=1, epochs=1)
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
    model = PaperLGSGMM(**model_config).to(config['device'])
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    (output_dir / f'{args.command}_config.json').write_text(json.dumps(config, indent=2) + '\n')
    if args.command == 'evaluate':
        if checkpoint is None:
            parser.error('evaluate requires --checkpoint')
        metrics, rows, second = evaluate(model, make_loader(config, args.split, args.limit), config,
                                         two_stage=not args.one_stage)
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
        restore_rng(checkpoint['rng'])
        start_epoch, best = checkpoint['epoch'] + 1, checkpoint['best_mda']
    history_path = output_dir / 'history.jsonl'
    if not args.resume and history_path.exists() and args.command != 'smoke':
        raise FileExistsError(f'Existing training run, use --resume or another --output-dir: {output_dir}')
    if args.command == 'smoke' and history_path.exists():
        history_path.unlink()
    limit = args.limit or (2 if args.command == 'smoke' else None)
    train_loader, val_loader = (make_loader(config, split, limit) for split in ('train', 'val'))
    for epoch in range(start_epoch, config['epochs']):
        training = train_epoch(model, train_loader, optimizer, scaler, config)
        state = rng_state()
        validation, _, _ = evaluate(model, val_loader, config, two_stage=args.command == 'smoke')
        restore_rng(state)
        scheduler.step(validation['loss'])
        mda = validation['one_stage']['sequence_macro']['mda']
        improved = mda > best
        best = max(best, mda)
        snapshot = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                        epoch=epoch, best_mda=best, config=config, rng=rng_state())
        atomic_save(snapshot, output_dir / 'last.pt')
        if improved:
            atomic_save(snapshot, output_dir / 'best.pt')
        record = dict(epoch=epoch + 1, learning_rate=optimizer.param_groups[0]['lr'],
                      train=training, validation=validation)
        with (output_dir / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record))
    if args.command == 'smoke':
        reloaded = torch.load(output_dir / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(reloaded['model'], strict=True)
        print('Smoke passed: train, backward, validation, second stage, checkpoint save/reload')


if __name__ == '__main__':
    main()
