import random
import copy
from sklearn.model_selection import ParameterGrid

from experiments.experiments import run_experiments

from experiments.opt_keys import *
from src.models.pooling.Model_opts import *

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
        opt_use_tokenclustering: False,
        opt_optimize_labels: True,
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

data_weights = {
    name_activity: 1,
    name_daytime: 1,
    name_loc: 1,
    name_timezone: 1,
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
    param_grid = generate_random_variants(seed=1, count_of_random_searches=500)
    run_experiments(param_grid, opts)