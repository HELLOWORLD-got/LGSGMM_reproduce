"""Rebuild identity-union manifests from raw MDMT XML and pinned official test TXT."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image


def read_xml(path):
    frames = defaultdict(dict)
    for track in ET.parse(path).getroot().findall('track'):
        identity = int(track.attrib['id'])
        for box in track.findall('box'):
            if box.attrib.get('outside', '0') == '1':
                continue
            coords = [float(box.attrib[k]) for k in ('xtl', 'ytl', 'xbr', 'ybr')]
            frames[int(box.attrib['frame'])][identity] = (coords, box.attrib.get('occluded', '0') == '1')
    return frames


def read_official(path):
    frames = defaultdict(dict)
    with path.open() as stream:
        for row in csv.reader(stream):
            frame, identity = int(row[0]) - 1, int(row[1]) - 1
            x, y, w, h = map(float, row[2:6])
            frames[frame][identity] = ([x, y, x + w, y + h], False)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='data/raw/MDMT')
    parser.add_argument('--official-test-dir', default='data/processed/MDMT/official_test')
    parser.add_argument('--output', default='data/processed/MDMT')
    args = parser.parse_args()
    raw, output = Path(args.raw).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test'):
        destination = output / (f'{split}.jsonl' if split != 'test' else 'test_official.jsonl')
        temporary = destination.with_suffix('.tmp')
        count = 0
        with temporary.open('w') as stream:
            directories = sorted((raw / split / '1').iterdir(), key=lambda p: int(p.name.split('-')[0]))
            for directory in directories:
                if not directory.is_dir():
                    continue
                sequence = directory.name.rsplit('-', 1)[0]
                images = []
                annotations = []
                for side in (1, 2):
                    name = f'{sequence}-{side}'
                    files = sorted((raw / split / str(side) / name).glob('*.jpg'), key=lambda p: int(p.stem))
                    images.append(files)
                    if split == 'test':
                        annotations.append(read_official(Path(args.official_test_dir) / f'{name}.txt'))
                    else:
                        annotations.append(read_xml(raw / 'new_xml' / str(side) / f'{name}.xml'))
                if len(images[0]) != len(images[1]):
                    raise ValueError(f'Unequal view frame counts: {sequence}')
                for frame, pair in enumerate(zip(*images)):
                    if pair[0].stem != pair[1].stem:
                        raise ValueError(f'Unsynchronized filenames: {pair}')
                    if split == 'test' and frame not in annotations[0] and frame not in annotations[1]:
                        continue
                    identities = sorted(set(annotations[0][frame]) | set(annotations[1][frame]))
                    boxes, visible, occluded, sizes = [], [], [], []
                    for side, path in enumerate(pair):
                        with Image.open(path) as image:
                            width, height = image.size
                        sizes.append([width, height])
                        view_boxes, view_visible, view_occluded = [], [], []
                        for identity in identities:
                            value = annotations[side][frame].get(identity)
                            valid = False
                            normalized = [0., 0., 0., 0.]
                            if value:
                                (x1, y1, x2, y2), occlusion = value
                                x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
                                y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
                                valid = x2 > x1 and y2 > y1
                                if valid:
                                    normalized = [(x1+x2)/(2*width), (y1+y2)/(2*height),
                                                  (x2-x1)/width, (y2-y1)/height]
                            view_boxes.append(normalized)
                            view_visible.append(valid)
                            view_occluded.append(bool(value and value[1]))
                        boxes.append(view_boxes)
                        visible.append(view_visible)
                        occluded.append(view_occluded)
                    row = dict(sequence=sequence, frame=frame, image_number=int(pair[0].stem),
                               images=[os.path.relpath(path, output) for path in pair], sizes=sizes,
                               ids=identities, labels=[0]*len(identities), boxes=boxes,
                               visible=visible, occluded=occluded)
                    stream.write(json.dumps(row, separators=(',', ':')) + '\n')
                    count += 1
        os.replace(temporary, destination)
        print(f'{destination}: {count} pairs')


if __name__ == '__main__':
    main()
