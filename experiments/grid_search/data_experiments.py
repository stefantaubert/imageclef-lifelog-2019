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

from src.data.Places365Data import name_places
from src.data.RawPlaces365Data import name_raw_places
from src.data.IndoorOutdoorData import name_io
from src.data.CocoYoloData import name_yolo
from src.data.CocoDetectronData import name_detectron
from src.data.CocoDefaultData import name_coco_default
from src.data.OpenImagesData import name_oi
from src.data.ImageNetData import name_imagenet
from src.data.ActivityData import name_activity
from src.data.DaytimeData import name_daytime
from src.data.LocationData import name_loc
from src.data.SUNattributesData import name_sun
from src.data.TimezoneData import name_timezone
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
        opt_comp_use_weights: True,
        opt_query_src: query_src_desc,
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

default_data_opts = {
    opt_weight: 1,
    opt_threshold: 0,
    opt_use_idf: False,
    opt_idf_boosting_threshold: 0,
    opt_intensify_factor_m: 1,
    opt_intensify_factor_p: 1,
    opt_ceiling: False,
}

def get_param_grid_df(param_grid):
    param_keys = []
    param_vals = []
    for param_settings in param_grid:
        ordered_param_settings = OrderedDict(sorted(param_settings.items(), key=lambda t: param_order[t[0]]))
        param_keys = list(ordered_param_settings.keys())
        param_vals.append(list(ordered_param_settings.values()))

    df = pd.DataFrame(param_vals, columns=param_keys)
    df = df.sort_values(param_keys, axis=0)
    return df

if __name__ == "__main__":
    ctx = ReadingContext(opts)
    m = ExperimentModel(ctx)

    seg_variants = [
        {
            opt_use_seg: [True],
            opt_img_selection: [img_selection_first, img_selection_all],
        },
        {
            opt_use_seg: [False],
            opt_img_selection: [img_selection_first],
        },
    ]

    idf_variants = [
        {
            opt_use_idf: [True],
            opt_idf_boosting_threshold: [0.5],
        },
        {
            opt_use_idf: [False],
            opt_idf_boosting_threshold: [0],
        },
    ]

    intensify_variants = [
        {
            opt_intensify_factor_m: [1],
            opt_intensify_factor_p: [1],
        },
        {
            opt_intensify_factor_m: [2],
            opt_intensify_factor_p: [2],
        },
        {
            opt_intensify_factor_m: [3],
            opt_intensify_factor_p: [3],
        },
    ]

    default_thresholds = [0, 0.9, 0.95, 0.99]
    places_thresholds = [0, 0.01, 0.05, 0.1]
    no_thresholds = [0]

    experiments = {
        # name_activity: no_thresholds,
        # name_daytime: no_thresholds,
        # name_loc: no_thresholds,
        # name_sun: no_thresholds,
        # name_timezone: no_thresholds,
        name_io: places_thresholds,
        name_raw_places: places_thresholds,
        # name_coco_default: default_thresholds,
        # name_detectron: default_thresholds,
        # name_imagenet: default_thresholds,
        # name_oi: default_thresholds,
        # name_yolo: default_thresholds,
        # name_places: places_thresholds,
    }

    for comparer, thresholds in experiments.items():
        print("Started experiments for", comparer)
        m.reset()
        opt_keys = []
        rows = []
        m.ctx.opts[opt_data].clear()
        m.ctx.opts[opt_data][comparer] = default_data_opts

        other_variants = [
            {
                opt_ceiling: [False],
                opt_threshold: thresholds,
                opt_comp_use_weights: [False],
                opt_comp_method: [comp_method_mean],
                opt_subm_imgs_per_day: [0, 1],
            }
        ]

        comparer_has_scores = len(thresholds) > 1

        if comparer_has_scores: 
            other_variants[0][opt_ceiling].append(True)

        param_grid = []
        for i in ParameterGrid(intensify_variants):
            for s in ParameterGrid(seg_variants):
                for p in ParameterGrid(idf_variants):
                    for o in ParameterGrid(other_variants):
                        params = dict()
                        params.update(i)
                        params.update(p)
                        params.update(s)
                        params.update(o)
                        param_grid.append(params)

        print(len(list(param_grid)))
        
        df = get_param_grid_df(param_grid)
        print(df)
        
        pbar = tqdm(total=df.shape[0])
        counter = 1
        for i, param_vals in df.iterrows():
            for param_key_i, value in enumerate(param_vals):
                m.change_opt(df.columns[param_key_i], value)
            print("Run experiment {}/{} with the following parameters:".format(str(counter), str(df.shape[0])))
            print(param_vals)
            subm = m.run()
            eval_res, eval_res_df = evaluate(subm, Xs=[5, 10, 20, 30, 40, 50])
            flattened_opts = flatten(m.ctx.opts)
            opt_keys = list(flattened_opts.keys())
            rows.extend([list(flattened_opts.values()) + eval_row for eval_row in eval_res])
            f1_10 = float(eval_res_df.loc[(eval_res_df['X'] == 10) & (eval_res_df['topic_id'] == 'all')]["F1@X"])
            print("F1@10:", str(f1_10))
            counter = counter + 1
            pbar.update(1)
            print("-------------")

        pbar.close()
        headers = opt_keys + ["topic_id", "X","CR@X", "P@X", "F1@X"]
        res = pd.DataFrame(columns=headers, data=rows)
        threshold_header = "{}_{}_{}".format(opt_data, comparer, opt_threshold)
        res = res.sort_values(by=[threshold_header, 'X', 'topic_id'])
        print(res)
        res.to_csv("/tmp/data_experiments_labelopt_{cmp}.csv".format(cmp=comparer), index=False)
