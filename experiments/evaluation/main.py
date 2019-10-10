from src.io.reader import read_gt
from experiments.evaluation.metrics import cr_x
from experiments.evaluation.metrics import p_x
from experiments.evaluation.metrics import f1_x
from experiments.evaluation.metrics import avg_cr_x
from experiments.evaluation.metrics import avg_p_x
from experiments.evaluation.metrics import avg_f1_x
from src.common.gt_converter import gt_to_dict
import pandas as pd

defaultXs = [0, 50, 40, 30, 20, 10, 5]

def evaluate_core_avg(top_subm, gt_dict, Xs = defaultXs):
    data = []

    for X in Xs:
        row = ['all', X]
        row.append(avg_cr_x(top_subm, gt_dict, X))
        row.append(avg_p_x(top_subm, gt_dict, X))
        row.append(avg_f1_x(top_subm, gt_dict, X))
        data.append(row)

    return data

def evaluate_core(top_subm, top_id, gt_dict, Xs = defaultXs):
    data = []

    for X in Xs:
        row = [X]
        row.append(cr_x(top_subm, gt_dict[top_id], X))
        row.append(p_x(top_subm, gt_dict[top_id], X))
        row.append(f1_x(top_subm, gt_dict[top_id], X))
        data.append(row)

    return data

def evaluate_top(top_subm, top_id, gt=read_gt(), Xs = defaultXs):
    gt_dict = gt_to_dict(gt)
    data = evaluate_core(top_subm, top_id, gt_dict, Xs)
    res = pd.DataFrame(columns=["X","CR@X", "P@X", "F1@X"], data=data)
    return res

def evaluate(subm, gt=read_gt(), Xs = defaultXs):
    gt_dict = gt_to_dict(gt)

    rows = []
    for top_id in list(gt_dict.keys()):
        data = evaluate_core(subm[top_id], top_id, gt_dict, Xs)
        data = [[top_id] + entry for entry in data]
        rows.extend(data)
    avg_data = evaluate_core_avg(subm, gt_dict, Xs)
    rows.extend(avg_data)
    res = pd.DataFrame(columns=["topic_id", "X","CR@X", "P@X", "F1@X"], data=rows)
    res = res.sort_values(by=['X', 'topic_id'])
    return rows, res
