"""Validate all supplied manifests and save portable, current-root audit results."""
import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/processed/MDMT')
    args = parser.parse_args()
    root = Path(args.data_dir).resolve()
    report = {}
    split_sequences = []
    for split in ('train', 'val', 'test'):
        sequences = set()
        keys = set()
        counts = Counter()
        dimensions = set()
        manifest = root / ('test_official.jsonl' if split == 'test' else f'{split}.jsonl')
        with manifest.open() as stream:
            for line in stream:
                row = json.loads(line)
                key = (row['sequence'], row['frame'])
                if key in keys:
                    raise ValueError(f'Duplicate pair: {split} {key}')
                keys.add(key)
                sequences.add(row['sequence'])
                if len(row['ids']) != len(set(row['ids'])):
                    raise ValueError(f'Duplicate identities: {key}')
                for side in range(2):
                    path = (root / row['images'][side]).resolve()
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    with Image.open(path) as image:
                        if list(image.size) != row['sizes'][side]:
                            raise ValueError(f'Wrong image size: {path}')
                        dimensions.add(image.size)
                    size = len(row['ids'])
                    for field in ('labels',):
                        if len(row[field]) != size:
                            raise ValueError(f'Length mismatch {field}: {key}')
                    for field in ('boxes', 'visible', 'occluded'):
                        if len(row[field][side]) != size:
                            raise ValueError(f'Length mismatch {field}: {key}')
                    for box, visible in zip(row['boxes'][side], row['visible'][side]):
                        if visible:
                            cx, cy, w, h = box
                            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                                raise ValueError(f'Invalid box: {key} {box}')
                            counts['visible_boxes'] += 1
                    counts['empty_views'] += not any(row['visible'][side])
                counts['pairs'] += 1
        split_sequences.append(sequences)
        report[split] = dict(counts, sequences=sorted(sequences, key=int),
                             sizes=sorted(dimensions))
    for i in range(3):
        for j in range(i + 1, 3):
            if split_sequences[i] & split_sequences[j]:
                raise ValueError('Sequence leakage across splits')
    report['protocol'] = 'Supplied class-agnostic GT boxes, per-sequence split, zero-based XML frame ordinal'
    target = root / 'lgsgmm_audit.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print(f'Saved {target}')


if __name__ == '__main__':
    main()
