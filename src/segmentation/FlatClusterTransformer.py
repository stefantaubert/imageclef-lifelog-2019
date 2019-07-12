from src.segmentation.CachableTransformerBase import CachableTransformerBase
import scipy.cluster.hierarchy as shc

class FlatClusterTransformer(CachableTransformerBase):
    """
    Apply the fcluster method to the given clusters.
    Parameters:
    - criterion: the criterion which taken for inter-cluster comparing and merging
    - threshold: the threshold for clusters which will be merged if the criterion <= threshold.
    Input: clusters (list of arrays)
    Output: flattend cluster ids as one-dim array (the length is the count of images in the input clusters, so that each image is assigned to one cluster)
    - Example: [368, 145, 366, 134, 485, 485, 485]
    """
    def __init__(self, day: int, threshold: int, usr: int, criterion='distance', is_dirty: bool = False):
        self.t = threshold
        self.c = criterion
        return super().__init__(usr=usr, is_dirty=is_dirty, suffix=str(day))

    def before_transform(self, _):
        print("Flattening clusters with criterion '{c}' and threshold '{t}'...".format(c=self.c, t=self.t))
    
    def transform_core(self, clusters: list):
        flat_clusters = shc.fcluster(clusters, t=self.t, criterion=self.c)
        return flat_clusters
    
    def after_transform(self, flat_clusters):
        count_clusters = len(set(flat_clusters))
        print("Count of clusters:", str(count_clusters))
