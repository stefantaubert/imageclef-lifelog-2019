import random
import copy
from sklearn.model_selection import ParameterGrid

from experiments.experiments import run_experiments

from experiments.opt_keys import *
from src.models.pooling.Model_opts import *

from src.data.RawPlaces365Data import name_raw_places
from src.data.IndoorOutdoorData import name_io
from src.data.CocoYoloData import name_yolo
from src.data.CocoDetectronData import name_detectron
from src.data.CocoDefaultData import name_coco_default
from src.data.OpenImagesData import name_oi
from src.data.ImageNetData import name_imagenet
from src.data.SUNattributesData import name_sun

opts = {

    opt_model: {
        opt_use_seg: True,
        opt_subm_imgs_per_day: 0,
        opt_subm_imgs_per_day_only_on_recall: False,
        opt_comp_method: comp_method_datamax,
        opt_comp_use_weights: True,
        opt_query_src: query_src_title,
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

    opt_data: {
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

        name_imagenet: {
            opt_weight: 1,
            opt_threshold: 0.99,
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
    },
}

data_weights = {
    name_coco_default: 1,
    name_detectron: 1,
    name_yolo: 1,
    name_imagenet: 1,
    name_oi: 1,
    name_raw_places: 1,
    name_io: 1,
    name_sun: 1,
}

def get_random_weight_variant():
    weights = copy.deepcopy(data_weights)
    for k in weights.keys():
        weights[k] = round(random.uniform(0, 1), 1)
    return weights

def generate_random_variants(seed, count_of_random_searches):
    random.seed(seed)
    variants = [{opt_data_weights: get_random_weight_variant()} for _ in range(count_of_random_searches)]
    return variants

if __name__ == "__main__":
    # made 100 experiments with seed 1
    param_grid = generate_random_variants(seed=2, count_of_random_searches=900)
    run_experiments(param_grid, opts)