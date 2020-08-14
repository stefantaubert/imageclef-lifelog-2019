from os.path import basename

from pdtransform import NoFitMixin
from src.io.paths import mk_dir
from src.common.helper import copy

def show_clusters_in_fs(clusters, dest_dir):
    origin_paths = []
    dest_paths = []

    for c in range(len(clusters)):
        s_count = len(clusters[c])

        for s in range(s_count):
            imgs = clusters[c][s]
            imgs_count = len(imgs)
            segment_dir = "{dir}{cluster}_{s_count}/{segment}_{img_count}/".format(dir=dest_dir, cluster=str(c), s_count=str(s_count), segment=str(s), img_count=str(imgs_count))
            
            for img in imgs:
                dest_file = "{segdir}{filename}".format(segdir=segment_dir, filename=basename(img))
                origin_paths.append(img)
                dest_paths.append(dest_file)
    
    copy(origin_paths, dest_paths)

class ShowInFsTransformer(NoFitMixin):
    """
    """
    def __init__(self, day: int, usr: int, name: str, copy: bool):
        self.day = day
        self.usr = usr
        self.copy = copy
        self.name = name
        
    def transform(self, clusters: list):
        if self.copy:
            print("Copying clusters to filesystem...")
            data_an_type = 1
            dest = mk_dir("cluster_output/{day}/{name}".format(day=str(self.day), name=self.name), usr=self.usr, t=data_an_type)
            show_clusters_in_fs(clusters, dest)
        return clusters