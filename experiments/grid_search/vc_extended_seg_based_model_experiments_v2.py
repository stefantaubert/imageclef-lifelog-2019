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
from src.data.SUNattributesData import name_sun
from experiments.helper import flatten

TO_BE_TESTED = None

opts = {

    opt_model: {
        opt_use_seg: True,
        opt_subm_imgs_per_day: TO_BE_TESTED,
        opt_subm_imgs_per_day_only_on_recall: TO_BE_TESTED,
        opt_comp_method: comp_method_datamax,
        opt_comp_use_weights: True,
        opt_query_src: query_src_title,
        opt_use_tokenclustering: True,
        opt_optimize_labels: True,
    },

    opt_tokenclustering: {
        opt_tokenclustering_comp_method: comp_method_clusters_mean,
        opt_tokenclustering_threshold: 0.5,
    },

    opt_segmentation: {
        opt_img_selection: TO_BE_TESTED,
        opt_s1_min_imgs: TO_BE_TESTED,
        opt_s1_threshold: TO_BE_TESTED,
        opt_merge_dist: 0,
        opt_repr_selection: repr_selection_first,
    },

    opt_data: {

        name_coco_default: {
            opt_weight: 0.1,
            opt_threshold: 0.9,
            opt_use_idf: False,
            opt_idf_boosting_threshold: 0,
            opt_intensify_factor_m: 1,
            opt_intensify_factor_p: 1,
            opt_ceiling: False,
        },

        name_detectron: {
            opt_weight: 0.7,
            opt_threshold: 0.95,
            opt_use_idf: False,
            opt_idf_boosting_threshold: 0,
            opt_intensify_factor_m: 1,
            opt_intensify_factor_p: 1,
            opt_ceiling: False,
        },

        name_yolo: {
            opt_weight: 0.3,
            opt_threshold: 0.9,
            opt_use_idf: False,
            opt_idf_boosting_threshold: 0,
            opt_intensify_factor_m: 1,
            opt_intensify_factor_p: 1,
            opt_ceiling: False,
        },

        name_imagenet: {
            opt_weight: 0.7,
            opt_threshold: 0.99,
            opt_use_idf: False,
            opt_idf_boosting_threshold: 0,
            opt_intensify_factor_m: 1,
            opt_intensify_factor_p: 1,
            opt_ceiling: False,
        },

        name_oi: {
            opt_weight: 0.2,
            opt_threshold: 0,
            opt_use_idf: True,
            opt_idf_boosting_threshold: 0.5,
            opt_intensify_factor_m: 2,
            opt_intensify_factor_p: 2,
            opt_ceiling: True,
        },
        
        name_raw_places: {
            opt_weight: 0.8,
            opt_threshold: 0,
            opt_use_idf: False,
            opt_idf_boosting_threshold: 0,
            opt_intensify_factor_m: 1,
            opt_intensify_factor_p: 1,
            opt_ceiling: True,
        },

        name_io: {
            opt_weight: 0.4,
            opt_threshold: 0,
            opt_use_idf: False,
            opt_idf_boosting_threshold: 0,
            opt_intensify_factor_m: 3,
            opt_intensify_factor_p: 3,
            opt_ceiling: False,
        },

        name_sun: {
            opt_weight: 0.8,
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
            opt_img_selection: [img_selection_first, img_selection_all],
            opt_s1_threshold: [23000, 26000, 41000, 50000],
            opt_s1_min_imgs: [1, 2],
        }
    ]

    param_grid = []
    for s in ParameterGrid(subm_variants):
        for o in ParameterGrid(other_variants):
            params = dict()
            params.update(s)
            params.update(o)
            param_grid.append(params)

    run_experiments(param_grid, opts)