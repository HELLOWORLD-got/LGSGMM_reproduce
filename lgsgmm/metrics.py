import numpy as np


def association_counts(views, matches):
    ids = [view['ids'].detach().cpu().tolist() for view in views]
    labels = [view['labels'].detach().cpu().tolist() for view in views]
    identities = [list(zip(label, identity)) for label, identity in zip(labels, ids)]
    gt = len(set(identities[0]) & set(identities[1]))
    pairs = np.asarray(matches, dtype=np.int64).reshape(-1, 2)
    correct = sum(identities[0][i] == identities[1][j] for i, j in pairs)
    return dict(tp=correct, fp=len(pairs) - correct, fn=gt - correct, gt=gt)


def scores(counts):
    tp, fp, fn, gt = (counts[key] for key in ('tp', 'fp', 'fn', 'gt'))
    return dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                mda=tp / max(gt + fp + fn, 1))


def summarize(rows):
    if not rows:
        raise ValueError('No evaluation rows')
    sequence_scores = {}
    for sequence in sorted({row['sequence'] for row in rows}):
        selected = [row for row in rows if row['sequence'] == sequence]
        sequence_scores[sequence] = {key: float(np.mean([scores(row)[key] for row in selected]))
                                     for key in ('precision', 'recall', 'mda')}
    totals = {key: sum(row[key] for row in rows) for key in ('tp', 'fp', 'fn', 'gt')}
    return dict(counts=totals, micro=scores(totals), per_sequence=sequence_scores,
                sequence_macro={key: float(np.mean([v[key] for v in sequence_scores.values()]))
                                for key in ('precision', 'recall', 'mda')},
                pair_macro={key: float(np.mean([scores(row)[key] for row in rows]))
                            for key in ('precision', 'recall', 'mda')})
