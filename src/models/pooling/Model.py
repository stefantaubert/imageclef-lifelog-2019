from src.io.ReadingContext import ReadingContext
from src.models.pooling.ImagePreprocessing import ImagePreprocessing
from src.models.pooling.SegmentPreprocessing import SegmentPreprocessing
from src.models.pooling.TokenPrediction import TokenPrediction
from src.models.pooling.ClusterPrediction import ClusterPrediction
from src.data.LocationData import LocationData
from src.data.DaytimeData import DaytimeData
from src.data.SUNattributesData import SUNattributesData
from src.data.Places365Data import Places365Data
from src.data.CocoDefaultData import CocoDefaultData
from src.data.CocoDetectronData import CocoDetectronData
from src.data.CocoYoloData import CocoYoloData
from src.data.OpenImagesData import OpenImagesData
from src.data.ActivityData import ActivityData
from src.data.ImageNetData import ImageNetData
from src.data.TimezoneData import TimezoneData
from src.data.RawPlaces365Data import RawPlaces365Data
from src.data.IndoorOutdoorData import IndoorOutdoorData
from src.models.pooling.VectorComparer import VectorComparer
from src.models.pooling.Postprocessing import Postprocessing
from src.models.pooling.Model_opts import *
from src.query_translation.get_query_tokens import get_query_tokens_from_topics

class Model():
    def __init__(self, opts: dict):
        assert opt_general in opts
        assert opt_usr in opts[opt_general]
        assert opt_ds in opts[opt_general]

        self.opts = opts
        self.ctx = ReadingContext(self.opts[opt_general][opt_usr], self.opts[opt_general][opt_ds])

    def fit_preprocessing(self):
        assert opt_model in self.opts
        assert opt_use_seg in self.opts[opt_model]
        
        if self.opts[opt_model][opt_use_seg]:
            prepr = SegmentPreprocessing(self.opts)
        else:
            prepr = ImagePreprocessing(self.opts[opt_general][opt_usr])
        
        self.imgs = prepr.images()
        self.repr_imgs = prepr.representatives()

        query_src = self.opts[opt_model][opt_query_src]
        ds = self.opts[opt_general][opt_ds]
        self.query_tokens = get_query_tokens_from_topics(self.ctx.tops(), query_src, ds, self.ctx.vocab())
        
    def fit_data_extraction(self):
        assert opt_data in self.opts

        data = [
            TimezoneData(self.ctx),
            CocoYoloData(self.ctx),
            OpenImagesData(self.ctx),
            CocoDetectronData(self.ctx),
            CocoDefaultData(self.ctx),
            SUNattributesData(self.ctx),
            Places365Data(self.ctx),
            DaytimeData(self.ctx),
            LocationData(self.ctx),
            ImageNetData(self.ctx),
            ActivityData(self.ctx),
            RawPlaces365Data(self.ctx),
            IndoorOutdoorData(self.ctx),
        ]

        self.data = []

        # TODO: parallel
        for d in data:
            data_name = d.get_name()

            if data_name not in self.opts[opt_data]:
                continue

            comparing_opts = self.opts[opt_data][data_name]
            weight = comparing_opts[opt_weight]

            if weight > 0:
                threshold = comparing_opts[opt_threshold]
                d.extract_data(threshold, self.opts[opt_model][opt_optimize_labels])
                self.data.append(d)
                
    def fit_comparers(self):
        vecs = []
        for data in self.data:
            comparer = VectorComparer(self.opts, self.ctx.emb(), self.imgs, self.query_tokens, data)
            comparer.fit()
            vecs.append(comparer)
        self.vecs = vecs

    def compare(self):
        for comparer in self.vecs:
            comparer.compare()

    def predict(self):
        if self.opts[opt_model][opt_use_tokenclustering]:
            p = ClusterPrediction(self.opts, self.ctx, self.data, self.vecs, self.repr_imgs, self.query_tokens)
        else:
            p = TokenPrediction(self.opts, self.ctx, self.data, self.vecs, self.repr_imgs, self.query_tokens)
        
        p.fit()
        self.sims = p.predict()
    
    def fit(self):
        self.fit_preprocessing()
        self.fit_data_extraction()
        self.fit_comparers()
        self.compare()

    def postprocess(self):
        self.postprocessing = Postprocessing(self.opts, self.sims, self.ctx.tops())
        self.postprocessing.fit()
        self.subm = self.postprocessing.process()

    def run(self):
        self.fit()
        self.predict()
        self.postprocess()
        return self.subm

