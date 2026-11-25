"""Score-guided EM, posterior filtering and homography-based supplementation."""
import cv2
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def weighted_similarity(x, y, posterior):
    mass = posterior.sum()
    if mass < 1e-12:
        return None
    wx, wy = posterior.sum(1), posterior.sum(0)
    mx, my = wx @ x / mass, wy @ y / mass
    xc, yc = x - mx, y - my
    covariance = xc.T @ posterior @ yc
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.ones(2)
    correction[-1] = np.linalg.det(u @ vt)
    rotation = u @ np.diag(correction) @ vt
    variance = (wx[:, None] * xc ** 2).sum()
    if variance < 1e-12:
        return None
    scale = (singular * correction).sum() / variance
    return scale * rotation, my - mx @ (scale * rotation)


def sgmm(x, y, scores, matches, alpha=.8, sigma2=1., uniform_density=1e-4,
         posterior_threshold=.5, iterations=50, tolerance=1e-5, **unused):
    m, n = len(x), len(y)
    if not m or not n:
        return np.empty((0, 2), dtype=np.int64), np.zeros((m, n))
    # Pixel coordinates normalized by joint spread keep sigma2=1 numerically meaningful.
    origin = np.vstack((x, y)).mean(0)
    spread = max(np.std(np.vstack((x, y))), 1.)
    a, b = (x - origin) / spread, (y - origin) / spread
    prior = np.full((m, n), 1 / m)
    for i, j in matches:
        score = np.clip(scores[i, j], 1e-8, 1 - 1e-8)
        prior[:, j] = (1 - score) / max(m - 1, 1)
        prior[i, j] = score if m > 1 else 1
    prior /= prior.sum(0, keepdims=True)
    gamma = float(np.clip(np.mean([scores[i, j] < .5 for i, j in matches])
                          if len(matches) else .5, .01, .99))
    transform, offset = np.eye(2), np.zeros(2)
    posterior = np.zeros((m, n))
    def expectation():
        distance = cdist(a @ transform + offset, b, 'sqeuclidean')
        # Eq. (19), evaluated in log-space to avoid underflow.
        log_weight = alpha * np.log(prior.clip(1e-12)) - distance / (2 * sigma2)
        normalizer = np.logaddexp(logsumexp(log_weight, axis=0), np.log(gamma * uniform_density))
        return np.exp(log_weight - normalizer[None])
    for _ in range(iterations):
        posterior = expectation()
        update = weighted_similarity(a, b, posterior)
        if update is None:
            break
        new_transform, new_offset = update
        change = np.linalg.norm(transform - new_transform) + np.linalg.norm(offset - new_offset)
        transform, offset = new_transform, new_offset
        distance = cdist(a @ transform + offset, b, 'sqeuclidean')
        sigma2 = max(float((posterior * distance).sum() / (2 * posterior.sum())), 1e-6)
        gamma = float(np.clip(1 - posterior.sum() / n, .001, .999))
        if change < tolerance:
            break
    posterior = expectation()
    # Enforce one-to-one correspondence after thresholding.
    candidates = np.argwhere(posterior > posterior_threshold)
    chosen, used_x, used_y = [], set(), set()
    for i, j in sorted(candidates.tolist(), key=lambda pair: posterior[tuple(pair)], reverse=True):
        if i not in used_x and j not in used_y:
            chosen.append((i, j))
            used_x.add(i)
            used_y.add(j)
    return np.asarray(chosen, dtype=np.int64).reshape(-1, 2), posterior


def refine(views, output, config):
    x, y = [view['centers'].detach().cpu().numpy().astype(np.float64) for view in views]
    initial = output['matches'].detach().cpu().numpy()
    scores = output['scores'].detach().cpu().numpy()
    filtered, posterior = sgmm(x, y, scores, initial, **config)
    if len(filtered) < 4:
        return filtered, dict(status='insufficient_inliers', homography=None)
    src, dst = x[filtered[:, 0]], y[filtered[:, 1]]
    if np.linalg.matrix_rank(src - src.mean(0)) < 2 or np.linalg.matrix_rank(dst - dst.mean(0)) < 2:
        return filtered, dict(status='degenerate_points', homography=None)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, config['ransac_threshold'])
    if homography is None or not np.isfinite(homography).all() or mask is None:
        return filtered, dict(status='homography_failed', homography=None)
    filtered = filtered[mask.ravel().astype(bool)]
    if len(filtered) < 4:
        return filtered, dict(status='insufficient_homography_inliers', homography=None)
    remaining_x = np.setdiff1d(np.arange(len(x)), filtered[:, 0])
    remaining_y = np.setdiff1d(np.arange(len(y)), filtered[:, 1])
    additions = []
    if len(remaining_x) and len(remaining_y):
        mapped = cv2.perspectiveTransform(x[remaining_x, None], homography)[:, 0]
        distances = cdist(mapped, y[remaining_y])
        threshold = config['mapping_threshold']
        # Invalid pairs cost more than any valid assignment.
        cost = np.where(np.isfinite(distances) & (distances <= threshold), distances,
                        (len(x) + len(y) + 1) * max(threshold, 1))
        rows, columns = linear_sum_assignment(cost)
        additions = [(remaining_x[i], remaining_y[j]) for i, j in zip(rows, columns)
                     if distances[i, j] <= threshold]
    combined = np.concatenate((filtered, np.asarray(additions, dtype=np.int64).reshape(-1, 2)))
    return combined, dict(status='ok', homography=homography.tolist(),
                         filtered=len(filtered), supplemented=len(additions))
