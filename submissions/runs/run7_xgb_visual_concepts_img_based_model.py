import sys
import os
import pickle
import pandas as pd

from experiments.evaluation.main import evaluate
from submissions.show_in_fs import show_in_fs_ctx

from src.io.ReadingContext import ReadingContext
from src.globals import ds_dev
from src.globals import ds_test
from src.globals import usr1
from src.models.xgb.XGBModel import XGBModel
from src.models.xgb.Model_opts import *
from src.models.pooling.Model_opts import *


from src.data.SUNattributesData import name_sun
from src.data.Places365Data import name_places
from src.data.CocoDefaultData import name_coco_default
from submissions.creator import create_submission

opts = {

    opt_general: {
        opt_usr: usr1,
        
        opt_mp: True,
        opt_ds: ds_dev,
    },

    opt_model: {
        opt_use_seg: False,
        opt_subm_imgs_per_day: 0,
        opt_subm_imgs_per_day_only_on_recall: False,
        opt_comp_method: comp_method_mean,
        opt_comp_use_weights: False,
        opt_query_src: query_src_desc,
        opt_use_tokenclustering: False,
        opt_optimize_labels: False,
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
        opt_predict_test: True,
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

def run():
    model_name = os.path.basename(sys.argv[0])[:-3]
    print("Started", model_name, "on testset.")
    xgb_model = XGBModel(ReadingContext(opts))
    xgb_model.run()
    val_subm = xgb_model.xgb_model.train_model.subm
    test_subm = xgb_model.xgb_model.test_model.subm

    _, eval_res_df = evaluate(val_subm, Xs=[5, 10, 20, 30, 40, 50])
    f1_10 = float(eval_res_df.loc[(eval_res_df['X'] == 10) & (eval_res_df['topic_id'] == 'all')]["F1@X"])
    print("F1@10:", str(f1_10))

    dir_dev = show_in_fs_ctx(val_subm, xgb_model.xgb_model.train_model.ctx, model_name, 20)
    print("Written top 20 images per dev topic to:", dir_dev)

    dir_test = show_in_fs_ctx(test_subm, xgb_model.xgb_model.test_model.ctx, model_name, 20)
    print("Written top 20 images per test topic to:", dir_test)

    file_name = create_submission(test_subm, "LMRT_TUC_MI_Stefan_Taubert_" + model_name)
    print("exported test submission to", file_name)

if __name__ == "__main__":
    run()