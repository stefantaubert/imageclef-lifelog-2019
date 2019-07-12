from src.segmentation.CachableTransformerBase import CachableTransformerBase
from src.io.paths import get_path_from_rel_path
from src.io.reader import read_vc

class VCReaderTransformer(CachableTransformerBase):
    """
    Reads the visual concepts database for the given user.
    Input: User id
    Output: Visual concepts database
    """
    def __init__(self, usr: int, is_dirty: bool = False):
        return super().__init__(usr=usr, is_dirty=is_dirty)
        
    def transform_core(self, _):
        vc = read_vc(usr=self.usr)
        return vc

if __name__ == "__main__":
    t = VCReaderTransformer(1)
    res = t.transform()
    print(res)
    