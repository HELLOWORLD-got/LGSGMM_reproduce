"""Regression checks for padding-free graphs, shared IDs and geometric refinement."""
import unittest

import numpy as np
import torch

from lgsgmm.loss import matching_loss
from lgsgmm.metrics import association_counts, scores
from lgsgmm.model import PaperLGSGMM, local_spectrum, mutual_matches
from lgsgmm.postprocess import refine, weighted_similarity


def view(points, identities):
    centers = torch.tensor(points, dtype=torch.float32).reshape(-1, 2)
    return dict(image=torch.zeros(3, 64, 64), centers=centers,
                positions=centers / 64,
                boxes=torch.cat((centers - 1, centers + 1), -1),
                ids=torch.tensor(identities), labels=torch.zeros(len(identities), dtype=torch.long))


class ReproductionTests(unittest.TestCase):
    def test_mda_penalizes_false_matches(self):
        views = [view([(3, 3), (5, 5)], [1, 2]), view([(3, 3), (5, 5), (7, 7)], [1, 2, 3])]
        counts = association_counts(views, [(0, 0), (1, 2)])
        self.assertEqual(scores(counts)['mda'], .25)

    def test_spectrum_small_graph(self):
        for count in (0, 1, 2, 6, 7):
            result = local_spectrum(torch.rand(count, 2), 6)
            self.assertEqual(tuple(result.shape), (count, 7))
            self.assertTrue(torch.isfinite(result).all())

    def test_empty_and_unequal_graphs_backward(self):
        model = PaperLGSGMM(layers=1, backbone_weights=None).eval()
        for counts in ((0, 0), (0, 2), (2, 1)):
            views = [view([(10 + i * 3, 10 + i * 2) for i in range(n)], list(range(n))) for n in counts]
            output = model(views)
            loss, _ = matching_loss(output, views)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            model.zero_grad(set_to_none=True)

    def test_geometric_refinement(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(12, 2)) * 100
        y = x @ np.array([[1.1, -.2], [.2, 1.1]]) + [50, 20]
        transform, translation = weighted_similarity(x, y, np.eye(12))
        np.testing.assert_allclose(x @ transform + translation, y, atol=1e-6)
        views = [dict(centers=torch.tensor(z)) for z in (x, y)]
        output = dict(matches=torch.tensor(np.column_stack((np.arange(12), np.arange(12)))),
                      scores=torch.eye(12) * .95)
        config = dict(alpha=.8, sigma2=1., uniform_density=1e-4, posterior_threshold=.5,
                      iterations=50, tolerance=1e-5, ransac_threshold=20., mapping_threshold=80.)
        matches, status = refine(views, output, config)
        self.assertEqual(status['status'], 'ok')
        self.assertEqual(len(matches), 12)
        self.assertTrue(np.all(matches[:, 0] == matches[:, 1]))

    def test_mutual_matching_one_to_one(self):
        pairs = mutual_matches(torch.tensor([[.9, .8], [.6, .7], [.1, .2]])).tolist()
        self.assertEqual(pairs, [[0, 0]])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
