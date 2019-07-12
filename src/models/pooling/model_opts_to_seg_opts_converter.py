from src.segmentation.SegmentationTransformer_opts import opt_usr
from src.segmentation.SegmentationTransformer_opts import opt_multiprocessing
from src.segmentation.SegmentationTransformer_opts import opt_s1_min_imgs
from src.segmentation.SegmentationTransformer_opts import opt_merge_dist
from src.segmentation.SegmentationTransformer_opts import opt_s1_threshold
from src.segmentation.SegmentationTransformer_opts import opt_soft_clear
from src.segmentation.SegmentationTransformer_opts import opt_clear_cache
from src.segmentation.SegmentationTransformer_opts import opt_copy_clusters
from src.segmentation.SegmentationTransformer_opts import opt_linkage_method
from src.segmentation.SegmentationTransformer_opts import opt_linkage_metric
from src.segmentation.SegmentationTransformer_opts import opt_fcluster_criterion
from src.segmentation.SegmentationTransformer_opts import linkage_method_average
from src.segmentation.SegmentationTransformer_opts import linkage_metric_euclidean
from src.segmentation.SegmentationTransformer_opts import fcluster_criterion_distance

from src.models.pooling.Model_opts import opt_general
from src.models.pooling.Model_opts import opt_usr as m_opt_usr
from src.models.pooling.Model_opts import opt_mp as m_opt_multiprocessing
from src.models.pooling.Model_opts import opt_segmentation
from src.models.pooling.Model_opts import opt_s1_min_imgs as m_opt_s1_min_imgs
from src.models.pooling.Model_opts import opt_merge_dist as m_opt_merge_dist
from src.models.pooling.Model_opts import opt_s1_threshold as m_opt_s1_threshold

def convert_to_seg_opts(model_opts: dict):
    assert opt_general in model_opts
    assert opt_segmentation in model_opts

    seg_opts = {
        opt_usr: model_opts[opt_general][m_opt_usr],
        opt_multiprocessing: model_opts[opt_general][m_opt_multiprocessing],
        opt_s1_min_imgs: model_opts[opt_segmentation][m_opt_s1_min_imgs],
        opt_merge_dist: model_opts[opt_segmentation][m_opt_merge_dist],
        opt_s1_threshold: model_opts[opt_segmentation][m_opt_s1_threshold],

        # non adjustable
        opt_soft_clear: True,
        opt_clear_cache: False,
        opt_copy_clusters: False,
        opt_linkage_method: linkage_method_average,
        opt_linkage_metric: linkage_metric_euclidean,
        opt_fcluster_criterion: fcluster_criterion_distance,
    }

    return seg_opts