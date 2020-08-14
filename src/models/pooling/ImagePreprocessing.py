from sklearn.pipeline import Pipeline

from src.io.ReadingContext import ReadingContext
from src.transformers.FlatteningTransformer import flatten
from src.segmentation.VCReaderTransformer import VCReaderTransformer
from src.segmentation.AutographerImageExtractorTransformer import AutographerImageExtractorTransformer
from src.segmentation.PathToIdTransformer import PathToIdTransformer
from src.transformers.WrappingTransformer import WrappingTransformer

from src.models.pooling.Model_opts import *

class ImagePreprocessing():
    def __init__(self, usr: int):
        self.usr = usr
    
    def __get_images__(self):
        pipe = Pipeline([
            ("read_vc", VCReaderTransformer(usr=self.usr, is_dirty=True)),
            ("extract_auto_imgs", AutographerImageExtractorTransformer(usr=self.usr, is_dirty=True)),
            ("wrap", WrappingTransformer()),
            ("paths_to_ids", PathToIdTransformer(-1, usr=self.usr, is_dirty=True)),
        ])

        self.imgs = pipe.transform([])
        self.repr = flatten(self.imgs)

    def images(self):
        try:
            return self.imgs
        except:
            self.__get_images__()
            return self.imgs
      
    def representatives(self):
        try:
            return self.repr
        except:
            self.__get_images__()
            return self.repr
        