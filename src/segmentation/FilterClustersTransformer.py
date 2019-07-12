from src.segmentation.CachableTransformerBase import CachableTransformerBase

def filter_clusters(clusters, min_img_count_per_segment):
    filtered = []

    for cluster in clusters:
        tmp_cluster = []
        for segment in cluster:
            if len(segment) >= min_img_count_per_segment:
                tmp_cluster.append(segment)
        if len(tmp_cluster) > 0:
            filtered.append(tmp_cluster)

    return filtered

class FilterClustersTransformer(CachableTransformerBase):
    """
    Filter the clusters, which have at least min_img_count images.
    Parameters:
    - min_img_count: minimal count of images which should be in a cluster
    Input: List of clusters with images as list
    Output: List of clusters with images as list but not including clusters with <= min_img_count images
    """
    def __init__(self, day: int, min_img_count: int, usr: int, is_dirty: bool = False):
        self.min_c = min_img_count
        return super().__init__(usr=usr, is_dirty=is_dirty, suffix=str(day))
        
    def before_transform(self, clusters):
        print("Removing segments which have < {c} images...".format(c=str(self.min_c)))
        self.old_segment_count = sum([len(c) for c in clusters])

    def transform_core(self, clusters: list):
        filtered = filter_clusters(clusters, self.min_c)
        return filtered

    def after_transform(self, filtered):
        segment_count = sum([len(c) for c in filtered])
        print("Segment count after filtering:", str(segment_count), "-> removed", str(self.old_segment_count-segment_count), "segments")
    