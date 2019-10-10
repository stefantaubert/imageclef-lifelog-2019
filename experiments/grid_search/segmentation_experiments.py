"Segmentation model is just for optimizing recall to 1.0"

from sklearn.model_selection import ParameterGrid
from experiments.evaluation.main import evaluate

from src.io.ReadingContext import ReadingContext
from src.globals import ds_dev
from src.globals import usr1
from src.models.pooling.Model import Model
from src.models.pooling.Model_opts import *

import pickle
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import ParameterGrid
from experiments.evaluation.main import evaluate
from submissions.show_in_fs import show_in_fs_ctx

from src.io.ReadingContext import ReadingContext
from src.globals import ds_dev
from src.globals import usr1
from experiments.ExperimentModel import ExperimentModel
from experiments.ExperimentModel import param_order
from collections import OrderedDict
from src.models.pooling.Model_opts import *

import os
import sys

from experiments.experiments import __get_param_grid_df__
from experiments.helper import flatten

opts = {

    opt_general: {
        opt_usr: usr1,
        
        opt_mp: True,
        opt_ds: ds_dev,
    },

    opt_model: {
        opt_use_seg: True,
        opt_subm_imgs_per_day: 0,
        opt_subm_imgs_per_day_only_on_recall: False,
        opt_comp_method: comp_method_mean,
        opt_comp_use_weights: False,
        opt_query_src: query_src_title,
        opt_use_tokenclustering: False,
        opt_optimize_labels: False,
    },

    opt_segmentation: {
        opt_img_selection: img_selection_first,
        opt_repr_selection: repr_selection_first,
        opt_s1_min_imgs: 2,
        opt_merge_dist: 15,
        opt_s1_threshold: 23000,
    },

    opt_data: { },
}

def run_experiments_recall(param_grid, default_settings, csv_suffix=""):

    model_name = os.path.basename(sys.argv[0])[:-3]
    print("Started", model_name)
    opts = dict(default_settings)
    ctx = ReadingContext(opts)
    m = ExperimentModel(ctx)
    m.reset()
    opt_keys = []
    rows = []

    print(len(list(param_grid)))
    
    df = __get_param_grid_df__(param_grid)
    print(df)

    current_max_cr = 0.0
    
    pbar = tqdm(total=df.shape[0])
    counter = 1
    for i, param_vals in df.iterrows():
        print("Run experiment {}/{} with the following parameters:".format(str(counter), str(df.shape[0])))
        print(param_vals)
        for param_key_i, value in enumerate(param_vals):
            m.change_opt(df.columns[param_key_i], value)
        subm = m.run()
        eval_res, eval_res_df = evaluate(subm, Xs=[0])
        cluster_recall = float(eval_res_df.loc[(eval_res_df['topic_id'] == 'all')]["CR@X"])
        print("Cluster Recall:", str(cluster_recall))
        flattened_opts = flatten(m.ctx.opts)
        opt_keys = list(flattened_opts.keys())
        rows.extend([[counter] + eval_row + list(flattened_opts.values()) for eval_row in eval_res])
        counter = counter + 1
        print("-------------")
        pbar.update(1)
        current_max_cr = max(current_max_cr, cluster_recall)
        print("Until now the best CR was:", str(current_max_cr))
        print("-------------")
    
    pbar.close()
    headers = ["run_id"] + ["topic_id", "X", "CR@X", "P@X", "F1@X"] + opt_keys
    res = pd.DataFrame(columns=headers, data=rows)
    res = res.sort_values(by=['X', 'topic_id'])
    print(res)
    res.to_csv("/tmp/{}{}.csv".format(model_name, csv_suffix), index=False)

if __name__ == "__main__":

    seg_variants = [
        {
            opt_s1_threshold: range(10000, 51000, 1000),
            opt_s1_min_imgs: range(2, 5),
            #opt_merge_dist: range(15, 50, 1),
            opt_repr_selection: [repr_selection_first, repr_selection_last],
        }
    ]

    # further optimization to find the best merge_distance
    seg_variants = [
        {
            opt_s1_threshold: range(37000, 51000, 4000),
            opt_s1_min_imgs: [2],
            opt_merge_dist: range(0, 30, 1),
            opt_repr_selection: [repr_selection_first],
        }
    ]

    param_grid = list(ParameterGrid(seg_variants))

    run_experiments_recall(param_grid, opts)
