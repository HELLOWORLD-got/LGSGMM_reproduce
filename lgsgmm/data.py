"""Read the existing MDMT identity-union JSONL manifests without copying images."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MDMTDataset(Dataset):
    def __init__(self, manifest, stride=1, limit=None, max_side=None):
        self.manifest = Path(manifest).resolve()
        if stride < 1:
            raise ValueError("stride must be positive")
        self.rows = []
        with self.manifest.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['frame'] % stride == 0:
                    self.rows.append(row)
        if limit is not None:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest}")
        self.max_side = max_side

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        views = []
        for side in range(2):
            path = (self.manifest.parent / row['images'][side]).resolve()
            with Image.open(path) as image:
                image = image.convert('RGB')
                if tuple(row['sizes'][side]) != image.size:
                    raise ValueError(f"Image dimensions disagree with manifest: {path}")
                if self.max_side and max(image.size) > self.max_side:
                    scale = self.max_side / max(image.size)
                    image = image.resize(tuple(max(1, round(n * scale)) for n in image.size))
                width, height = image.size
                tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
            visible = torch.tensor(row['visible'][side], dtype=torch.bool)
            boxes = torch.tensor(row['boxes'][side], dtype=torch.float32).reshape(-1, 4)[visible]
            ids = torch.tensor(row['ids'], dtype=torch.long)[visible]
            labels = torch.tensor(row['labels'], dtype=torch.long)[visible]
            centers = boxes[:, :2] * torch.tensor([width, height])
            half = boxes[:, 2:] * torch.tensor([width, height]) / 2
            xyxy = torch.cat((centers - half, centers + half), dim=-1)
            xyxy[:, 0::2].clamp_(0, width)
            xyxy[:, 1::2].clamp_(0, height)
            views.append(dict(image=tensor, boxes=xyxy, centers=centers,
                              positions=boxes[:, :2], ids=ids, labels=labels))
        return dict(sequence=row['sequence'], frame=row['frame'], views=views)


def collate_pairs(samples):
    # Keep each graph at its actual size, including empty views.
    return samples


def to_device(sample, device):
    return [dict((key, value.to(device)) for key, value in view.items())
            for view in sample['views']]
