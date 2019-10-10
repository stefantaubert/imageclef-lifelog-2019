from sklearn.model_selection import ParameterGrid

from experiments.experiments import run_experiments

from src.models.pooling.Model_opts import *

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

opts = {
    opt_model: {
        opt_use_seg: True,
        opt_subm_imgs_per_day: 0,
        opt_subm_imgs_per_day_only_on_recall: False,
        opt_comp_method: comp_method_mean,
        opt_comp_use_weights: False,
        opt_query_src: query_src_desc,
        opt_use_tokenclustering: False,
        opt_optimize_labels: True,
    },

    opt_segmentation: {
        opt_img_selection: img_selection_first,
        opt_repr_selection: repr_selection_first,
        opt_s1_min_imgs: 2,
        opt_merge_dist: 15,
        opt_s1_threshold: 23000,
    },

    opt_tokenclustering: {
        opt_tokenclustering_comp_method: comp_method_clusters_mean,
        opt_tokenclustering_threshold: 0.5,
    },

    opt_data: {

        name_activity: {
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

        name_detectron: {
            opt_weight: 1,
            opt_threshold: 0.95,
            opt_use_idf: False,
            opt_idf_boosting_threshold: 0,
            opt_intensify_factor_m: 1,
            opt_intensify_factor_p: 1,
            opt_ceiling: False,
        },

        name_yolo: {
            opt_weight: 1,
            opt_threshold: 0.9,
            opt_use_idf: False,
            opt_idf_boosting_threshold: 0,
            opt_intensify_factor_m: 1,
            opt_intensify_factor_p: 1,
            opt_ceiling: False,
        },

        name_daytime: {
            opt_weight: 1,
            opt_threshold: 0,
            opt_use_idf: False,
            opt_idf_boosting_threshold: 0,
            opt_intensify_factor_m: 1,
            opt_intensify_factor_p: 1,
            opt_ceiling: False,
        },

        name_imagenet: {
            opt_weight: 1,
            opt_threshold: 0.99,
            opt_use_idf: False,
            opt_idf_boosting_threshold: 0,
            opt_intensify_factor_m: 1,
            opt_intensify_factor_p: 1,
            opt_ceiling: False,
        },

        name_loc: {
            opt_weight: 1,
            opt_threshold: 0,
            opt_use_idf: False,
            opt_idf_boosting_threshold: 0,
            opt_intensify_factor_m: 1,
            opt_intensify_factor_p: 1,
            opt_ceiling: False,
        },

        name_oi: {
            opt_weight: 1,
            opt_threshold: 0,
            opt_use_idf: True,
            opt_idf_boosting_threshold: 0.5,
            opt_intensify_factor_m: 2,
            opt_intensify_factor_p: 2,
            opt_ceiling: True,
        },
        
        name_raw_places: {
            opt_weight: 1,
            opt_threshold: 0,
            opt_use_idf: False,
            opt_idf_boosting_threshold: 0,
            opt_intensify_factor_m: 1,
            opt_intensify_factor_p: 1,
            opt_ceiling: True,
        },

        name_io: {
            opt_weight: 1,
            opt_threshold: 0,
            opt_use_idf: False,
            opt_idf_boosting_threshold: 0,
            opt_intensify_factor_m: 3,
            opt_intensify_factor_p: 3,
            opt_ceiling: False,
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

        name_timezone: {
            opt_weight: 1,
            opt_threshold: 0,
            opt_use_idf: False,
            opt_idf_boosting_threshold: 0,
            opt_intensify_factor_m: 1,
            opt_intensify_factor_p: 1,
            opt_ceiling: False,
        },
    },
}

if __name__ == "__main__":
    tokenclustering_variants = [
        {
            opt_use_tokenclustering: [False],
            opt_tokenclustering_comp_method: [comp_method_clusters_mean],
        },
        {
            opt_use_tokenclustering: [True],
            opt_tokenclustering_comp_method: [comp_method_clusters_mean, comp_method_clusters_max],
        }
    ]

    subm_variants = [
        {
            opt_subm_imgs_per_day_only_on_recall: [False],
            opt_subm_imgs_per_day: [0, 1],
        },
        {
            opt_subm_imgs_per_day_only_on_recall: [True],
            opt_subm_imgs_per_day: [1],
        }
    ]

    other_variants = [
        {
            opt_query_src: [query_src_title, query_src_desc],
            opt_comp_use_weights: [False, True],
            opt_comp_method: [comp_method_mean, comp_method_datamax, comp_method_tokenmax],
            opt_img_selection: [img_selection_first, img_selection_all],
        }
    ]

    param_grid = []
    for t in ParameterGrid(tokenclustering_variants):
        for s in ParameterGrid(subm_variants):
            for o in ParameterGrid(other_variants):
                params = dict()
                params.update(t)
                params.update(s)
                params.update(o)
                param_grid.append(params)

    run_experiments(param_grid, opts)