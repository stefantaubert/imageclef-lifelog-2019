from src.segmentation.CachableTransformerBase import CachableTransformerBase
import scipy.cluster.hierarchy as shc

class ClusterTransformer(CachableTransformerBase):
    """
    Perform hierarchical (agglomerative) clustering to the given histograms.
    Parameters:
    - metric: the metric which is used for comparing the histograms.
    - method: the method which is used for inter-cluster comparison.
    Input: list of histograms
    Output: cluster obtained through linkage
    """
    def __init__(self, day: int, usr: int, is_dirty: bool = False, metric='euclidean', method='average'):
        self.metric = metric
        self.method = method
        return super().__init__(usr=usr, is_dirty=is_dirty, suffix=str(day))
        
    def before_transform(self, _):
        print("Building clusters with metric '{metric}' and method '{method}'...".format(metric=self.metric, method=self.method))

    def transform_core(self, histograms: list):
        cluster = shc.linkage(histograms, method=self.method, metric=self.metric)
        return cluster