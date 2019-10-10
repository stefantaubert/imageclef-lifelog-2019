import sys
import os
import pickle
import pandas as pd
import copy

from experiments.evaluation.main import evaluate
from submissions.show_in_fs import show_in_fs_ctx

from src.io.ReadingContext import ReadingContext
from src.globals import ds_dev
from src.globals import ds_test
from src.globals import usr1
from src.models.pooling.xgb_model.XGBModel import XGBModel
from src.models.pooling.Model_opts import *


from src.data.SUNattributesData import name_sun
from src.data.Places365Data import name_places
from src.data.CocoDefaultData import name_coco_default
from src.submissions.creator import create_submission

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
        opt_query_src: query_src_desc,
        opt_use_tokenclustering: False,
        opt_optimize_labels: False,
    },

    opt_segmentation: {
        opt_img_selection: img_selection_all,
        opt_repr_selection: repr_selection_first,
        opt_s1_min_imgs: 2,
        opt_merge_dist: 15,
        opt_s1_threshold: 23000,
    },

    opt_data: {

        name_places: {
            opt_weight: 1,
            opt_threshold: 0,
            opt_use_idf: False,
            opt_idf_boosting_threshold: 0,
            opt_intensify_factor_m: 1,
            opt_intensify_factor_p: 1,
            opt_ceiling: True,
        },

        name_sun: {
            opt_weight: 1,
            opt_threshold: 0,
            opt_use_idf: False,
            opt_idf_boosting_threshold: 0,
            opt_intensify_factor_m: 1,
            opt_intensify_factor_p: 1,
            opt_ceiling: False,
        },
        
        name_coco_default: {
            opt_weight: 1,
            opt_threshold: 0.9,
            opt_use_idf: False,
            opt_idf_boosting_threshold: 0,
            opt_intensify_factor_m: 1,
            opt_intensify_factor_p: 1,
            opt_ceiling: False,
        },
    },

    opt_xgb: {
        opt_cut_negatives: 0,
        opt_validation_size_negatives: 0.2,
        opt_validation_size_positives: 0.2,
        opt_predict_test: False,
        opt_num_boost_round: 500,
        opt_early_stopping_rounds: 10,
        opt_verbose_eval: True,
        opt_predictor: predictor_gpu,
        opt_tree_method: tree_method_gpu_hist,
        opt_eval_metric: eval_metric_logloss,
        opt_objective: objective_bin_logistic,
        opt_max_depth: 3,
        opt_silent: 0,
        opt_seed: 4242,
    },
}

def run_xgb(opts):
    xgb_model = XGBModel(Context(opts))
    xgb_model.run()
    val_subm = xgb_model.xgb_model.train_model.ctx.mem[mem_subm]

    _, eval_res_df = evaluate(val_subm, Xs=[5, 10, 20, 30, 40, 50])
    f1_10 = float(eval_res_df.loc[(eval_res_df['X'] == 10) & (eval_res_df['topic_id'] == 'all')]["F1@X"])
    return f1_10

def run_max_dept_exp():
    print("Running max depth exp...")
    depts = [1,2,3,4]
    for d in depts:
        run1_opts = copy.deepcopy(opts)
        run1_opts[opt_xgb][opt_max_depth] = d
        f1 = run_xgb(run1_opts)
        print("maxdepth {}: F1@10:".format(str(d)), str(f1))

def run_sorted_exp():
    print("Running sorted exp...")
    run1_opts = copy.deepcopy(opts)
    run1_opts[opt_xgb][opt_max_depth] = 3
    run1_opts[opt_model][opt_subm_imgs_per_day] = 1
    f1 = run_xgb(run1_opts)
    print("sorted: F1@10:", str(f1))

def run():
    model_name = os.path.basename(sys.argv[0])[:-3]
    print("Started", model_name, "on devset.")
    
    #run_max_dept_exp()
    run_sorted_exp()

if __name__ == "__main__":
    run()