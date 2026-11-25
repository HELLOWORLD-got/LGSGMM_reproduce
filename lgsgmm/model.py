"""Equations (1)-(11), with explicit choices for unspecified architecture details."""
from pathlib import Path
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights

from zz.all_nn.roi_pooling import CustomRoIPooling


def local_spectrum(points, neighbors):
    """All e+1 eigenvalues of each normalized local graph Laplacian."""
    count = len(points)
    result = points.new_zeros((count, neighbors + 1))
    if count < 2:
        return result
    # CPU eigendecomposition also supports systems without accelerator eigh.
    coords = points.detach().cpu().double()
    distance = torch.cdist(coords, coords)
    distance.fill_diagonal_(float('inf'))
    k = min(neighbors, count - 1)
    nearest = distance.topk(k, largest=False).indices
    indices = torch.cat((torch.arange(count)[:, None], nearest), dim=1)
    groups = coords[indices]
    local_distance = torch.cdist(groups, groups)
    rho = local_distance[:, 0, 1:].mean(dim=1).clamp_min(1e-6)
    adjacency = torch.exp(-local_distance.square() / rho[:, None, None].square())
    eye = torch.eye(k + 1)[None]
    adjacency = adjacency * (1 - eye)
    degree = adjacency.sum(dim=-1).clamp_min(1e-12).rsqrt()
    laplacian = eye - degree[:, :, None] * adjacency * degree[:, None, :]
    spectrum = torch.linalg.eigvalsh(laplacian).float().to(points.device)
    result[:, :k + 1] = spectrum
    return result


def position_distance(points):
    count = len(points)
    if count <= 1:
        return points.new_ones((count, count))
    distance = torch.cdist(points, points)
    diagonal = torch.eye(count, dtype=torch.bool, device=points.device)
    inverse = distance.clamp_min(1e-6).reciprocal().masked_fill(diagonal, 0)
    inverse = inverse / inverse[~diagonal].mean().clamp_min(1e-6)
    # The paper leaves the singular self-distance unspecified.
    return inverse.masked_fill(diagonal, 1)


class GuidedAttention(nn.Module):
    def __init__(self, dimension, heads):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.ffn = nn.Sequential(nn.Linear(dimension, dimension * 2), nn.LayerNorm(dimension * 2), nn.GELU(),
                                 nn.Linear(dimension * 2, dimension))

    def forward(self, source, target, source_position, target_position, pdm=None):
        if len(source) == 0 or len(target) == 0:
            return source
        def split(value):
            return value.reshape(len(value), self.heads, -1).transpose(0, 1)
        query = split(self.query(source + source_position))
        key = split(self.key(target + target_position))
        value = split(self.value(target))
        weights = (query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])).softmax(-1)
        if pdm is not None:
            weights = weights * pdm[None]
        message = (weights @ value).transpose(0, 1).reshape_as(source)
        return source + self.ffn(message)


class EnhancementLayer(nn.Module):
    def __init__(self, dimension, heads):
        super().__init__()
        self.self_attention = GuidedAttention(dimension, heads)
        self.cross_attention = GuidedAttention(dimension, heads)

    def forward(self, a, b, positions, spectra, distances, pga, lga):
        a = self.self_attention(a, a, positions[0], positions[0], distances[0] if pga else None)
        b = self.self_attention(b, b, positions[1], positions[1], distances[1] if pga else None)
        fa, fb = spectra if lga else (torch.zeros_like(a), torch.zeros_like(b))
        # Both directions use the same pre-cross-attention features.
        return (self.cross_attention(a, b, fa, fb),
                self.cross_attention(b, a, fb, fa))


class PaperLGSGMM(nn.Module):
    def __init__(self, descriptor_dim=256, layers=7, heads=4, neighbors=6,
                 pga=True, lga=True, backbone_weights='DEFAULT',
                 imagenet_normalize=True, match_threshold=0.0):
        super().__init__()
        if descriptor_dim % heads:
            raise ValueError('descriptor_dim must be divisible by heads')
        weights = ResNet50_Weights.DEFAULT if backbone_weights == 'DEFAULT' else None
        backbone = resnet50(weights=weights)
        if backbone_weights and backbone_weights != 'DEFAULT':
            path = Path(backbone_weights)
            if path.read_bytes()[:70].startswith(b'version https://git-lfs'):
                raise ValueError(f'Git LFS pointer, download real weights first: {path}')
            backbone.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.roi = CustomRoIPooling(output_dim=2048)
        self.projection = nn.Linear(2048, descriptor_dim)
        self.position = nn.Sequential(nn.Linear(2, descriptor_dim), nn.GELU(),
                                      nn.Linear(descriptor_dim, descriptor_dim))
        self.spectral_projection = nn.Linear(neighbors + 1, descriptor_dim)
        self.layers = nn.ModuleList(EnhancementLayer(descriptor_dim, heads) for _ in range(layers))
        self.neighbors, self.pga, self.lga = neighbors, pga, lga
        self.imagenet_normalize = imagenet_normalize
        self.match_threshold = match_threshold
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, views):
        features, positions, spectra, distances = [], [], [], []
        for view in views:
            image = view['image'][None]
            if self.imagenet_normalize:
                image = (image - self.mean) / self.std
            # Empty sets still permit classification/triplet losses to return graph-connected zero.
            if len(view['boxes']):
                feature_map = self.backbone(image)
                boxes = view['boxes']
                xywh = torch.cat((boxes[:, :2], boxes[:, 2:] - boxes[:, :2]), -1)[None]
                mask = torch.ones((1, len(boxes)), dtype=torch.bool, device=image.device)
                pooled = self.roi(feature_map, xywh, mask, image.shape[-2], image.shape[-1])[0]
                features.append(self.projection(pooled))
            else:
                features.append(self.projection.weight.sum(dim=1)[None, :][:0])
            positions.append(self.position(view['positions'] * 2 - 1))
            spectra.append(self.spectral_projection(local_spectrum(view['positions'], self.neighbors)))
            distances.append(position_distance(view['positions']))
        a, b = features
        for layer in self.layers:
            a, b = layer(a, b, positions, spectra, distances, self.pga, self.lga)
        similarity = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T
        if len(a) and len(b):
            scores = similarity.softmax(0) * similarity.softmax(1)
        else:
            scores = similarity
        matches = mutual_matches(scores, self.match_threshold)
        return dict(features=(a, b), scores=scores, matches=matches)


def mutual_matches(scores, threshold=0.0):
    if min(scores.shape) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=scores.device)
    best_b = scores.argmax(1)
    indices = torch.arange(len(scores), device=scores.device)
    keep = (scores.argmax(0)[best_b] == indices) & (scores[indices, best_b] > threshold)
    return torch.stack((indices[keep], best_b[keep]), dim=1)
