from src.segmentation.CachableTransformerBase import CachableTransformerBase
from src.io.img_path_parser import get_paths_cached

class PathToIdTransformer(CachableTransformerBase):
    """
    Maps the imagepaths of the segments to the corresponding image_ids because it is more practical to use the image ids.
    Input: segments with imagepats
    Output: segments with imageids
    """
    def __init__(self, day: int, usr: int, is_dirty: bool = False):
        return super().__init__(usr=usr, is_dirty=is_dirty, suffix=str(day))

    def before_transform(self, _):
        print("Getting image ids for paths...")
    
    def transform_core(self, segments: list):
        paths_to_ids = get_paths_cached(usr=self.usr)
        ids_to_paths = {y:x for x,y in paths_to_ids.items()}
        for segment in segments:
            for i in range(len(segment)):
                segment[i] = ids_to_paths[segment[i]]
        return segments
    
    def after_transform(self, _):
        #print(self.result)
        pass
