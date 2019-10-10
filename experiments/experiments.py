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

from experiments.helper import flatten

gen_opts = {
    opt_general: {
        opt_usr: usr1,
        opt_mp: True,
        opt_ds: ds_dev,
    }
}

def __get_param_grid_df__(param_grid):
    param_keys = []
    param_vals = []
    for param_settings in param_grid:
        ordered_param_settings = OrderedDict(sorted(param_settings.items(), key=lambda t: param_order[t[0]]))
        param_keys = list(ordered_param_settings.keys())
        param_vals.append(list(ordered_param_settings.values()))

    df = pd.DataFrame(param_vals, columns=param_keys)
    try:
        # at random search it is not possible to sort because dict items cannot be sorted
        df = df.sort_values(param_keys, axis=0)
    except: pass
    return df

def run_experiments(param_grid, default_settings, csv_suffix=""):

    model_name = os.path.basename(sys.argv[0])[:-3]
    print("Started", model_name)
    opts = dict(default_settings)
    opts.update(gen_opts)
    #ctx = ReadingContext(opts[opt_general][opt_usr], opts[opt_general][opt_ds])
    m = ExperimentModel(opts)
    m.reset()
    opt_keys = []
    rows = []

    print(len(list(param_grid)))
    
    df = __get_param_grid_df__(param_grid)
    print(df)

    current_max_f1 = 0.0
    
    pbar = tqdm(total=df.shape[0])
    counter = 1
    for i, param_vals in df.iterrows():
        print("Run experiment {}/{} with the following parameters:".format(str(counter), str(df.shape[0])))
        print(param_vals)
        for param_key_i, value in enumerate(param_vals):
            m.change_opt(df.columns[param_key_i], value)
        subm = m.run()
        eval_res, eval_res_df = evaluate(subm, Xs=[1, 5, 10, 20, 30, 40, 50])
        f1_10 = float(eval_res_df.loc[(eval_res_df['X'] == 10) & (eval_res_df['topic_id'] == 'all')]["F1@X"])
        print("F1@10:", str(f1_10))
        flattened_opts = flatten(m.opts)
        opt_keys = list(flattened_opts.keys())
        rows.extend([[counter] + eval_row + list(flattened_opts.values()) for eval_row in eval_res])
        counter = counter + 1
        print("-------------")
        pbar.update(1)
        current_max_f1 = max(current_max_f1, f1_10)
        print("Until now the best F1@10 was:", str(current_max_f1))
        print("-------------")
    
    pbar.close()
    headers = ["run_id"] + ["topic_id", "X", "CR@X", "P@X", "F1@X"] + opt_keys
    res = pd.DataFrame(columns=headers, data=rows)
    res = res.sort_values(by=['X', 'topic_id'])
    print(res)
    res.to_csv("/tmp/{}{}.csv".format(model_name, csv_suffix), index=False)
