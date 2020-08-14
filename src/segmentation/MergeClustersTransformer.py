from src.segmentation.CachableTransformerBase import CachableTransformerBase
from src.segmentation.DaySelectorTransformer import DaySelectorTransformer
from src.segmentation.FilterClustersTransformer import filter_clusters

def merge_segments(clusters, imgs, max_segment_distance):
    for cluster in clusters:
        s_count = len(cluster)
        has_at_least_2_segments = s_count >= 2
        if has_at_least_2_segments:
            for s in range(s_count - 1):
                current_segment = cluster[s]
                next_segment = cluster[s + 1]
                last_from_current = current_segment[-1]
                first_from_next = next_segment[0]
                pos_last = imgs.index(last_from_current)
                pos_first = imgs.index(first_from_next)
                assert pos_last < pos_first
                count_of_imgs_between = pos_first - pos_last - 1
                if count_of_imgs_between <= max_segment_distance:
                    cluster[s + 1] = current_segment + next_segment
                    cluster[s] = []

    clusters = filter_clusters(clusters, 1)
    return clusters

class MergeClustersTransformer(CachableTransformerBase):
    """
    Merge segments together whose last und first image is maximal n images apart.
    Parameters:
    - max_segment_distance: maximal distance between segment 1 and segment 2 for merging those segments into one
    Input: list of clusters with segments
    Output: list of clusters with merged segments if possible
    """
    def __init__(self, day: int, max_segment_distance: int, usr: int, is_dirty: bool = False):
        self.day = day
        self.max_d = max_segment_distance
        return super().__init__(usr=usr, is_dirty=is_dirty, suffix=str(day))
        
    def before_transform(self, _):
        print("Merge segments which are only <= {c} images apart...".format(c=str(self.max_d)))

    def transform_core(self, clusters: list):
        imgs = DaySelectorTransformer(self.day, usr=self.usr).from_cache()
        merged = merge_segments(clusters, imgs, self.max_d)
        return merged
    
    def after_transform(self, merged):
        segment_count = sum([len(c) for c in merged])
        print("Segment count after merging:", str(segment_count))