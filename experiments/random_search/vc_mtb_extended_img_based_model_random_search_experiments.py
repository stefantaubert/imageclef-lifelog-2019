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
from src.data.ActivityData import name_activity
from src.data.DaytimeData import name_daytime
from src.data.LocationData import name_loc
from src.data.TimezoneData import name_timezone

opts = {

    opt_model: {
        opt_use_seg: False,
        opt_subm_imgs_per_day: 0,
        opt_subm_imgs_per_day_only_on_recall: False,
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

        name_daytime: {
            opt_weight: 1,
            opt_threshold: 0,
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
        
    },
}

best_vc_extended_weights = {
    name_coco_default: 0.1,
    name_detectron: 0.7,
    name_yolo: 0.3,
    name_imagenet: 0.7,
    name_oi: 0.2,
    name_raw_places: 0.8,
    name_io: 0.4,
    name_sun: 0.8,
}

mtb_weights = {
    name_activity: 1,
    name_daytime: 1,
    name_loc: 1,
    name_timezone: 1,
}

def get_random_mtb_weight_variant():
    weights: dict = copy.deepcopy(mtb_weights)

    for k in weights.keys():
        weights[k] = round(random.uniform(0, 1), 1)
    
    weights.update(copy.deepcopy(best_vc_extended_weights))

    return weights

def generate_random_variants(seed, count_of_random_searches):
    random.seed(seed)
    variants = [{opt_data_weights: get_random_mtb_weight_variant()} for _ in range(count_of_random_searches)]
    all_one = copy.deepcopy(mtb_weights)
    all_one.update(copy.deepcopy(best_vc_extended_weights))
    variants.insert(0, {opt_data_weights: all_one})

    all_zero = copy.deepcopy(mtb_weights)
    for k in all_zero.keys(): all_zero[k] = 0
    all_zero.update(copy.deepcopy(best_vc_extended_weights))
    variants.insert(0, {opt_data_weights: all_zero})

    return variants

if __name__ == "__main__":
    param_grid = generate_random_variants(seed=1, count_of_random_searches=100)
    run_experiments(param_grid, opts)