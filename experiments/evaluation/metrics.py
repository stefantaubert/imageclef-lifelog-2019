from collections import OrderedDict

def p_x(top_subm: OrderedDict, top_gt: dict, x: int):
    """Calculates the Precision @ X.

    Keyword arguments:
    top_subm -- dictionary, where keys = predicted image_ids and values = confidence_scores for the topic
    top_gt -- ground truth, where keys = cluster_id and values = possible image_ids for the topic
    x -- the top X results of top_subm will be counted

    Remarks:
    x == 0 means Precision without X, so it counts all predictions
    scores are irrelevant
    """
    assert isinstance(top_subm, OrderedDict)
    assert isinstance(top_gt, dict)
    assert isinstance(x, int)
    assert x >= 0
    assert top_gt

    if not top_subm:
        return 0
    
    pred = list(top_subm.keys())
    if x == 0:
        top_pred = pred
        x = len(top_pred)
    else:
        top_pred = pred[:x]
    
    hits = 0
    all_clusters = top_gt.keys()

    for pred_id in top_pred:
        for cluster_id in all_clusters:
            current_cluster_ids = top_gt[cluster_id]
            cluster_was_predicted = pred_id in current_cluster_ids

            if cluster_was_predicted:
                hits += 1
                break

    pred_len = len(top_pred)

    return hits / pred_len

def cr_x(top_subm: OrderedDict, top_gt: dict, x: int):
    """Calculates the Cluster Recall @ X.

    Keyword arguments:
    top_subm -- dictionary, where keys = predicted image_ids and values = confidence_scores for the topic
    top_gt -- ground truth, where keys = cluster_id and values = possible image_ids for the topic
    x -- the top X results of top_subm will be counted

    Remarks:
    x == 0 means Cluster Recall without X, so it counts all predictions
    if x is smaller than the cluster count at one topic the topic count is the limit
    scores are irrelevant
    """
    assert isinstance(top_subm, OrderedDict)
    assert isinstance(top_gt, dict)
    assert isinstance(x, int)
    assert x >= 0
    assert top_gt

    if not top_subm:
        return 0

    pred = list(top_subm.keys())
    if x == 0:
        top_pred = pred
        x = len(top_pred)
    else:
        top_pred = pred[:x]

    hits = 0
    all_clusters = top_gt.keys()

    for cluster_id in all_clusters:
        current_cluster_ids = top_gt[cluster_id]

        for pred_id in top_pred:
            cluster_was_predicted = pred_id in current_cluster_ids
            if cluster_was_predicted:
                hits += 1
                break

    cluster_len = len(top_gt.keys())
    
    # min(hits, x) makes sense if >=2 clusters contains >1 same image_ids and those are a hit
    # min(cluster_len, x) because it doesn't plays a role which clusters are hitted (cluster order is irrelevant)
    score = min(hits, x) / min(cluster_len, x)

    return score

def f1_x(top_subm: OrderedDict, top_gt: dict, x: int):
    """Calculates the F1 @ X.

    Keyword arguments:
    top_subm -- dictionary, where keys = predicted image_ids and values = confidence_scores for the topic
    top_gt -- ground truth, where keys = cluster_id and values = possible image_ids for the topic
    x -- the top X results of top_subm will be counted

    Remarks:
    x == 0 means F1 without X, so it counts all predictions
    scores are irrelevant
    """
    assert isinstance(top_subm, OrderedDict)
    assert isinstance(top_gt, dict)
    assert isinstance(x, int)
    assert x >= 0
    assert top_gt

    cr = cr_x(top_subm, top_gt, x)
    p = p_x(top_subm, top_gt, x)

    if p == 0 and cr == 0:
        return 0
    else:
        score = 2 * (p * cr) / (p + cr)
        return score

def avg_score(subm, gt, x, metric):
    scores = []

    for top in gt.keys():
        has_predictions_for_top = top in subm.keys()

        if has_predictions_for_top:
            score = metric(subm[top], gt[top], x)
            scores.append(score)
        else:
            scores.append(0)
    
    avg = sum(scores) / len(scores)
    
    return avg

def avg_f1_x(subm, gt, x):
    # Note: the averages of cr@x and p@x can both together be higher than the average f1@x!
    return avg_score(subm, gt, x, f1_x)

def avg_cr_x(subm, gt, x):
    return avg_score(subm, gt, x, cr_x)

def avg_p_x(subm, gt, x):
    return avg_score(subm, gt, x, p_x)
