from src.models.pooling.Model_opts import *

from src.data.Places365Data import name_places
from submissions.runs.run_base import run_on_dev
from submissions.runs.run_base import run_on_test

opts = {
    opt_model: {
        opt_query_src: query_src_desc,
        opt_use_seg: False,
        opt_subm_imgs_per_day: 0,
        opt_subm_imgs_per_day_only_on_recall: False,
        opt_comp_method: comp_method_mean,
        opt_comp_use_weights: False,
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
    },
}

if __name__ == "__main__":
    #run_on_dev(opts)
    run_on_test(opts)
