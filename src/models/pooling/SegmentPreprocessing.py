from src.io.ReadingContext import ReadingContext

from sklearn.pipeline import Pipeline

from src.transformers.FlatteningTransformer import FlatteningTransformer
from src.transformers.RepresentativeImageTransformer import RepresentativeImageTransformer
from src.segmentation.SegmentationTransformer import SegmentationTransformer

from src.models.pooling.Model_opts import *
from src.models.pooling.model_opts_to_seg_opts_converter import convert_to_seg_opts

class SegmentPreprocessing():
    def __init__(self, opts: dict):
        self.opts = opts
    
    def __get_segments__(self):
        assert opt_general in self.opts
        assert opt_usr in self.opts[opt_general]
        assert opt_segmentation in self.opts
        assert opt_img_selection in self.opts[opt_segmentation]
        assert opt_repr_selection in self.opts[opt_segmentation]

        pipe = Pipeline([
            ("get_segments", SegmentationTransformer(convert_to_seg_opts(self.opts))),
            ("flatten_days_to_segments", FlatteningTransformer()),
        ])

        img_pipe = Pipeline([
            ("select_segment_imgs", RepresentativeImageTransformer(self.opts[opt_general][opt_usr], method=self.opts[opt_segmentation][opt_img_selection], is_dirty=True)),
        ])

        repr_pipe = Pipeline([
            ("select_representative_imgs", RepresentativeImageTransformer(self.opts[opt_general][opt_usr], method=self.opts[opt_segmentation][opt_repr_selection], is_dirty=True)),
            ("flatten_segments", FlatteningTransformer()),
        ])

        segments = pipe.transform([])

        self.imgs = img_pipe.transform(segments)
        self.repr = repr_pipe.transform(segments)
        
    def images(self):
        try:
            return self.imgs
        except:
            self.__get_segments__()
            return self.imgs

    def representatives(self):
        try:
            return self.repr
        except:
            self.__get_segments__()
            return self.repr
        