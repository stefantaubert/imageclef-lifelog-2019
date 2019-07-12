from src.comparing.get_similarities_opts import opt_use_idf
from src.comparing.get_similarities_opts import opt_idf_boosting_threshold
from src.comparing.get_similarities_opts import opt_intensify_factor_m
from src.comparing.get_similarities_opts import opt_intensify_factor_p
from src.comparing.get_similarities_opts import opt_ceiling
from src.comparing.get_similarities_opts import opt_multiprocessing

from src.models.pooling.Model_opts import opt_general
from src.models.pooling.Model_opts import opt_data
from src.models.pooling.Model_opts import opt_mp as m_opt_multiprocessing
from src.models.pooling.Model_opts import opt_use_idf as m_opt_use_idf
from src.models.pooling.Model_opts import opt_idf_boosting_threshold as m_opt_idf_boosting_threshold
from src.models.pooling.Model_opts import opt_intensify_factor_m as m_opt_intensify_factor_m
from src.models.pooling.Model_opts import opt_intensify_factor_p as m_opt_intensify_factor_p
from src.models.pooling.Model_opts import opt_ceiling as m_opt_ceiling

def convert_to_comparing_opts(model_opts: dict, data_key: str):
    assert opt_general in model_opts
    assert opt_data in model_opts

    seg_opts = {
        opt_multiprocessing: model_opts[opt_general][m_opt_multiprocessing],
        opt_use_idf: model_opts[opt_data][data_key][m_opt_use_idf],
        opt_idf_boosting_threshold: model_opts[opt_data][data_key][m_opt_idf_boosting_threshold],
        opt_intensify_factor_m: model_opts[opt_data][data_key][m_opt_intensify_factor_m],
        opt_intensify_factor_p: model_opts[opt_data][data_key][m_opt_intensify_factor_p],
        opt_ceiling: model_opts[opt_data][data_key][m_opt_ceiling],
    }

    return seg_opts