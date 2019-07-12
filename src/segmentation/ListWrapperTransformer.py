from src.segmentation.CachableTransformerBase import CachableTransformerBase

class ListWrapperTransformer(CachableTransformerBase):
    """
    Wapping the segments of a day into a seperate list, so that the merging process of featureuniun returns a list of segments for each day
    Input: list of segments
    Output: list of segments wrappen in a list
    """
    def __init__(self, day: int, usr: int, is_dirty: bool = False):
        self.day = day
        return super().__init__(usr=usr, is_dirty=is_dirty, suffix=str(day))

    def transform_core(self, segments: list):
        return { self.day: segments }
    
    def after_transform(self, _):
        #print(self.result)
        pass
