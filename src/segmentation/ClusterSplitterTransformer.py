from src.segmentation.CachableTransformerBase import CachableTransformerBase
from src.segmentation.DaySelectorTransformer import DaySelectorTransformer

def build_clusters(imgs, segments):
    assert len(imgs) == len(segments)
    
    # no double entries
    assert len(imgs) == len(set(imgs))
    
    # segments must be 1-based
    assert 0 not in segments

    cluster_count = len(set(segments))
    clusters = [[] for c in range(cluster_count)]
    
    prev_cluster_id = -1

    for i in range(len(imgs)):
        img = imgs[i]
        cluster_id = segments[i] - 1
        cluster = clusters[cluster_id]
        if prev_cluster_id == cluster_id:
            cluster[-1].append(img)
        else:
            cluster.append([img])
        prev_cluster_id = cluster_id
    
    return clusters

class ClusterSplitterTransformer(CachableTransformerBase):
    """
    Assigns different cluster ids to images which have the same cluster id but were seperated through other images from an other cluster
    Input: list of cluster ids for the images.
    - Example: [1,1,2,2,1,1,3,1]
    Output: list of clusters with imagepaths as list in segments.
    - Example: [[[1,1],[3,3],[5]],[[2,2]],[[4]]]
    """
    def __init__(self, day: int, usr: int, is_dirty: bool = False):
        self.day = day
        return super().__init__(usr=usr, is_dirty=is_dirty, suffix=str(day))
    
    def before_transform(self, _):
        print("Extracting segment parts...")

    def transform_core(self, flat_clusters: list):
        imgs = DaySelectorTransformer(self.day, usr=self.usr).from_cache()
        clusters = build_clusters(imgs, flat_clusters)
        return clusters
        
    def after_transform(self, clusters):
        segment_count = sum([len(c) for c in clusters])
        print("Segment count:", str(segment_count))
    