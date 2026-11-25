import torch
from torch.nn import functional as F


def matching_loss(output, views, classification_weight=0.6, triplet_weight=0.3, margin=1.0):
    a, b = output['features']
    zero = a.sum() * 0 + b.sum() * 0
    if not len(a) or not len(b):
        return zero, dict(classification=0.0, triplet=0.0)
    truth = (views[0]['ids'][:, None] == views[1]['ids'][None])
    truth = truth & (views[0]['labels'][:, None] == views[1]['labels'][None])
    # Full score supervision also covers true associations missed by discrete matching.
    classification = F.binary_cross_entropy(output['scores'].clamp(1e-7, 1 - 1e-7), truth.float())
    positive_pairs = truth.nonzero()
    terms = []
    for i, j in positive_pairs:
        negatives = (~truth[i]).nonzero().flatten()
        if len(negatives):
            k = negatives[torch.randint(len(negatives), (), device=a.device)]
            positive_distance = (a[i] - b[j]).square().sum()
            negative_distance = (a[i] - b[k]).square().sum()
            terms.append((positive_distance - negative_distance + margin).clamp_min(0))
    triplet = torch.stack(terms).mean() if terms else zero
    total = classification_weight * classification + triplet_weight * triplet
    return total, dict(classification=float(classification.detach()), triplet=float(triplet.detach()))
