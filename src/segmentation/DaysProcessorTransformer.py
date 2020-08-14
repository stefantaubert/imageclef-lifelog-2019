from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from src.segmentation.CachableTransformerBase import CachableTransformerBase
from src.segmentation.DaySelectorTransformer import DaySelectorTransformer
from src.segmentation.HistogramTransformer import HistogramTransformer
from src.segmentation.ClusterTransformer import ClusterTransformer
from src.segmentation.FlatClusterTransformer import FlatClusterTransformer
from src.segmentation.ClusterSplitterTransformer import ClusterSplitterTransformer
from src.segmentation.FilterClustersTransformer import FilterClustersTransformer
from src.segmentation.MergeClustersTransformer import MergeClustersTransformer
from src.segmentation.PathToIdTransformer import PathToIdTransformer
from src.segmentation.ListWrapperTransformer import ListWrapperTransformer
from src.segmentation.ShowInFsTransformer import ShowInFsTransformer
from src.transformers.FlatteningTransformer import FlatteningTransformer
from src.segmentation.SegmentationTransformer_opts import *

class DaysProcessorTransformer(CachableTransformerBase):
    """
    Process all segmentation steps per day.
    Input: images of a day as list for all days
    - Example: [
        [day1_img_path1, day1_img_path2, ...],
        [day2_img_path1001, day2_img_path1002, ...],
    ]
    Output: segments of a day (image ids for each segment) for all days as a list
    - Example: [
        [day1_seg1, day1_seg2, day1_seg3, ...],
        [day2_seg1, day2_seg2, day2_seg3, ...],
        ...
    ] 
    where day1_seg1 is: [
        img_id1, img_id2, ...
    ]
    """
    def __init__(self, settings: dict):
        self.setts = settings
        self.soft_clr = self.setts[opt_soft_clear]
        return super().__init__(usr=self.setts[opt_usr], is_dirty=True)
        
    def before_transform(self, _):
        print("Processing days...")
    
    def transform_core(self, days: list):
        day_pipes = []

        for i in range(len(days)):
            day_str = "day_{i}".format(i=str(i+1))
            day_pipe = (day_str, Pipeline([
                ("select_" + day_str, DaySelectorTransformer(i, usr=self.usr)),
                ("calc_histogram_" + day_str, HistogramTransformer(i, usr=self.usr)),
                ("build_clusters_" + day_str, ClusterTransformer(i, usr=self.usr, metric=self.setts[opt_linkage_metric], method=self.setts[opt_linkage_method], is_dirty=self.soft_clr)),
                ("flattening_clusters_" + day_str, FlatClusterTransformer(i, threshold=self.setts[opt_s1_threshold], criterion=self.setts[opt_fcluster_criterion], usr=self.usr, is_dirty=self.soft_clr)),
                ("building_segments" + day_str, ClusterSplitterTransformer(i, usr=self.usr, is_dirty=self.soft_clr)),
                ("filter_segments" + day_str, FilterClustersTransformer(i, min_img_count=self.setts[opt_s1_min_imgs], usr=self.usr, is_dirty=self.soft_clr)),
                ("output_fs_filtered" + day_str, ShowInFsTransformer(i, name="filtered", usr=self.usr, copy=self.setts[opt_copy_clusters])),
                ("merge_segments" + day_str, MergeClustersTransformer(i, max_segment_distance=self.setts[opt_merge_dist], usr=self.usr, is_dirty=self.soft_clr)),
                ("output_fs_merged" + day_str, ShowInFsTransformer(i, name="merged", usr=self.usr, copy=self.setts[opt_copy_clusters])),
                ("extract_segments" + day_str, FlatteningTransformer()),
                ("paths_to_ids" + day_str, PathToIdTransformer(i, usr=self.usr, is_dirty=self.soft_clr)),
                ("wrap_to_list" + day_str, ListWrapperTransformer(i, usr=self.usr, is_dirty=self.soft_clr)),
            ]))

            day_pipes.append(day_pipe)
            
        job_count = -1 if self.setts[opt_multiprocessing] else 1
        tmp = FeatureUnion(day_pipes, n_jobs=job_count)
        result = tmp.transform(days)
        # this is necessary to unpack the wrapped segments
        result = [list(r.values())[0] for r in result]
        return result
