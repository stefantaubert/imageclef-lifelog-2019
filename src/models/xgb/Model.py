import numpy as np
import random
import xgboost as xgb

from tqdm import tqdm

from src.models.pooling.Model import Model as PoolingModel
from src.io.ReadingContext import ReadingContext
from src.models.pooling.ImagePreprocessing import ImagePreprocessing
from src.models.pooling.SegmentPreprocessing import SegmentPreprocessing
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
from src.models.xgb.XGBTokenPrediction import XGBTokenPrediction
from src.models.xgb.Model_opts import *
from src.models.xgb.TrainEvalSetCreator import TrainEvalSetCreator

class Model():

    def __init__(self, opts: dict, fitted_train_model: PoolingModel, fitted_test_model: PoolingModel):
        self.opts = opts

        np.random.seed(self.opts[opt_xgb][opt_seed])
        random.seed(self.opts[opt_xgb][opt_seed])

        self.train_model = fitted_train_model
        self.test_model = fitted_test_model
        self.ds_creator = TrainEvalSetCreator(self.ctx, self.train_model, self.test_model)

    def __get_concatenated_train_set__(self):
        x_train = []
        y_train = []

        trn_sets: dict = self.ds_creator.trainset
        trn_sets_keys = sorted(trn_sets.keys())
        for trn_set_key in trn_sets_keys: 
            _, x, y = trn_sets[trn_set_key]
            x_train.extend(x)
            y_train.extend(y)

        return x_train, y_train
    
    def __get_concatenated_validation_set__(self):
        x_valid = []
        y_valid = []

        val_sets: dict = self.ds_creator.validset
        val_sets_keys = sorted(val_sets.keys())
        for val_set_key in val_sets_keys: 
            _, x, y = val_sets[val_set_key]
            x_valid.extend(x)
            y_valid.extend(y)
        
        return x_valid, y_valid

    def __train_model__(self):
        x_valid, y_valid = self.__get_concatenated_validation_set__()
        x_train, y_train = self.__get_concatenated_train_set__()
        
        d_valid = xgb.DMatrix(x_valid, label=y_valid)
        d_train = xgb.DMatrix(x_train, label=y_train)

        watchlist = [(d_train, "train"), (d_valid, "valid")]
        self.bst = xgb.train(
            params=self.ctx.opts[opt_xgb],
            dtrain=d_train,
            num_boost_round=self.ctx.opts[opt_xgb][opt_num_boost_round], 
            early_stopping_rounds=self.ctx.opts[opt_xgb][opt_early_stopping_rounds],
            verbose_eval=self.ctx.opts[opt_xgb][opt_verbose_eval],
            evals=watchlist,
        )

    def __predict_validationset__(self):
        val_res = {}
        for i in tqdm(range(len(self.ctx.tops()))):
            top_imgs, top_x, _ = self.ds_creator.validset[i]
            d_valid = xgb.DMatrix(top_x)
            top_valid_pred = self.bst.predict(d_valid, ntree_limit=self.bst.best_ntree_limit)
            val_res[i] = {}
            for j in range(len(top_imgs)):
                img_repr = top_imgs[j]
                val_res[i][img_repr] = top_valid_pred[j]

        self.train_model.ctx.mem[mem_sims] = val_res

    def __predict_testset__(self):
        test_res = {}
        reps = self.test_model.repr_imgs
        for i in tqdm(range(len(self.test_model.ctx.tops()))):
            top_x = self.ds_creator.testset[i]
            d_test = xgb.DMatrix(top_x)
            top_pred = self.bst.predict(d_test, ntree_limit=self.bst.best_ntree_limit)
            test_res[i] = {}
            for img_i in range(len(reps)):
                img_repr = reps[img_i]
                test_res[i][img_repr] = top_pred[img_i]
    
        self.test_model.ctx.mem[mem_sims] = test_res

    def fit(self):
        print("Comparing dev...")
        p = XGBTokenPrediction(self.train_model.ctx)
        p.fit()
        p.predict()

        if self.ctx.opts[opt_xgb][opt_predict_test]:
            print("Comparing test...")
            p = XGBTokenPrediction(self.test_model.ctx)
            p.fit()
            p.predict()

        print("Building datasets...")
        self.ds_creator.fit()

    def train(self):
        print("Training model...")
        self.__train_model__()

    def predict(self):
        print("Predicting validationset...")
        self.__predict_validationset__()
        self.train_model.postprocess()

        if self.ctx.opts[opt_xgb][opt_predict_test]:
            print("Predicting testset...")
            self.__predict_testset__()
            self.test_model.postprocess()
            test_subm = self.test_model.subm
        
        return self.train_model.subm
