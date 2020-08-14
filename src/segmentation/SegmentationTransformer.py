import time

from sklearn.pipeline import Pipeline

from src.segmentation.VCReaderTransformer import VCReaderTransformer
from src.segmentation.AutographerImageExtractorTransformer import AutographerImageExtractorTransformer
from src.segmentation.DaysExtractionTransformer import DaysExtractionTransformer
from src.segmentation.DaysProcessorTransformer import DaysProcessorTransformer
from src.segmentation.CachableTransformerBase import CachableTransformerBase
from src.common.helper import clear_cache
from src.segmentation.SegmentationTransformer_opts import *

class SegmentationTransformer(CachableTransformerBase):
    """
    Performs the segmentation process.
    Input: user id
    Output: for each day a list of contained segments in which were the contained image ids as list
    """
    def __init__(self, opts: dict):
        self.opts = opts
        return super().__init__(usr=self.opts[opt_usr], is_dirty=True)

    def before_transform(self, _):
        self.start = time.time()
        print("Perform segmentation...")
        if self.opts[opt_clear_cache]:
            print("Cleaning cache...")
            clear_cache(self.usr)

    def transform_core(self, _):
        pipe = Pipeline([
            ("read_vc", VCReaderTransformer(usr=self.usr)),
            ("extract_auto_imgs", AutographerImageExtractorTransformer(usr=self.usr)),
            ("extract_days", DaysExtractionTransformer(usr=self.usr)),
            ("segment_days", DaysProcessorTransformer(settings=self.opts)),
        ])

        result = pipe.transform([])
        return result
    
    def get_file_count(self, days):
        c = 0
        for segments in days:
            for segment in segments:
                c += len(segment)
        return c

    def after_transform(self, days):
        original_files_count = len(AutographerImageExtractorTransformer(usr=self.usr).from_cache())
        segment_count = sum([len(segments) for segments in days])
        image_count = self.get_file_count(days)
        print("Original image count:", str(original_files_count))
        print("Final image count:", str(image_count))
        print("Reduzed image count by:", str(round(100-image_count/original_files_count*100,0)) + "%")
        print("Final segment count:", str(segment_count))
        mins = int(round((time.time() - self.start) / 60.0, 0))
        print("Total duration of segmentation process:", str(mins), "minutes")
