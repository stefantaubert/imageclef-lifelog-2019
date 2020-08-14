from src.segmentation.CachableTransformerBase import CachableTransformerBase

class DaySelectorTransformer(CachableTransformerBase):
    """
    Select the day of the given list.
    Input: list of days
    Output: Images of a day
    """
    def __init__(self, day, usr: int, is_dirty: bool = False):
        self.idx = day
        return super().__init__(usr=usr, is_dirty=is_dirty, suffix=str(day))
    
    def before_transform(self, X):
        print("Start day", str(self.idx))
        
    def transform_core(self, days):
        return days[self.idx]