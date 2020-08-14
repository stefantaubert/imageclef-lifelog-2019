from pdtransform import NoFitMixin
from src.io.paths import get_path_from_rel_path
from src.io.reader import read_vc
from tqdm import tqdm
from src.segmentation.CachableTransformerBase import CachableTransformerBase
from pandas import DataFrame

def extract_autograph_filenames(image_ids, image_paths, usr=1):
    assert len(image_ids) == len(image_paths)
    paths = []
    for i in tqdm(range(len(image_ids))):
        img_id = image_ids[i]
        img_path = image_paths[i]
        is_personal = "cam" in img_id
        if is_personal:
            continue
        else:
            full_path = get_path_from_rel_path(img_path, is_cam=False, usr=usr)
            paths.append(full_path)
    return paths

class AutographerImageExtractorTransformer(CachableTransformerBase):
    """
    Extracts all paths to autographer images from the given visual concept database.
    Input: Visual concepts database as DataFrame
    Output: List of paths to all autographer images which exist for the user
    """
    def __init__(self, usr: int, is_dirty: bool=False):
        return super().__init__(usr=usr, is_dirty=is_dirty)
        
    def before_transform(self, _):
        print("Getting autographer images...")

    def transform_core(self, vc: DataFrame):
        img_ids = list(vc["image_id"])
        img_paths = list(vc["image_path"])
        filenames = extract_autograph_filenames(img_ids, img_paths, self.usr)
        return filenames

    def after_transform(self, filenames):
        print("Extracted {c} images for user {u}.".format(c=str(len(filenames)), u=str(self.usr)))

if __name__ == "__main__":
    from sklearn.pipeline import Pipeline
    from src.segmentation.VCReaderTransformer import VCReaderTransformer

    usr = 1
    pipe = Pipeline([
        ("read_vc", VCReaderTransformer(skip=False)),
        ("extract_auto_imgs", AutographerImageExtractorTransformer(usr, is_dirty=True))
    ])
    
    res = pipe.transform(usr)
    print(len(res))