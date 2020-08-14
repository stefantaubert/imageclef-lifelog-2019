from src.query_translation.get_clusters_opts import opt_linkage_method
from src.query_translation.get_clusters_opts import opt_linkage_metric 
from src.query_translation.get_clusters_opts import opt_fcluster_criterion
from src.query_translation.get_clusters_opts import opt_fcluster_threshold
from src.query_translation.get_clusters_opts import fcluster_criterion_distance
from src.query_translation.get_clusters_opts import linkage_method_single
from src.query_translation.get_clusters_opts import linkage_metric_cosine

from src.models.pooling.Model_opts import opt_tokenclustering
from src.models.pooling.Model_opts import opt_tokenclustering_threshold

def convert_to_token_clustering_opts(opts: dict):
    assert opt_tokenclustering in opts
    assert opt_tokenclustering_threshold in opts[opt_tokenclustering]

    opts = {
        opt_linkage_method: linkage_method_single,
        opt_linkage_metric: linkage_metric_cosine,
        opt_fcluster_criterion: fcluster_criterion_distance,
        opt_fcluster_threshold: opts[opt_tokenclustering][opt_tokenclustering_threshold]
    }

    return opts