import numpy as np
from sklearn.metrics import ndcg_score


def tf_ndcg(y_true, y_pred, y_group, top_n=5):
    """Compute mean NDCG@k grouped by query."""
    begin_idx = 0
    ndcg_scores = []
    for g in y_group:
        y_t = y_true[begin_idx:begin_idx + g]
        y_p = y_pred[begin_idx:begin_idx + g]
        if g > 0:
            ndcg_scores.append(ndcg_score([y_t], [y_p], k=min(top_n, g)))
        begin_idx += g
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
