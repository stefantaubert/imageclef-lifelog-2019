import numpy as np

from tqdm import tqdm
from math import ceil
import random

from src.io.ReadingContext import ReadingContext
from src.models.pooling.Model import Model as PoolingModel
from src.common.train_test_split import train_test_split_gt
from src.common.train_test_split import train_test_split_list
from src.globals import top_idi
from src.models.pooling.Model_opts import *
from src.models.xgb.Model_opts import *

class TrainEvalSetCreator():
    """
    Parameters ctx:
    - cut_negatives
    - train_size
    - train_size_gt

    Parameters test_ctx:
    - predict
    """
    def __init__(self, ctx: ReadingContext, opts: dict, fitted_train_model: PoolingModel, fitted_test_model: PoolingModel):

        self.ctx = ctx
        self.opts = opts
        self.train_model = fitted_train_model
        self.test_model = fitted_test_model
    
    def __build_testset__(self):
        sims = self.test_model.sims
        reps = self.test_model.repr_imgs
        res = {}
        for top_i in tqdm(range(len(self.test_model.ctx.tops()))):
            x_test = []
            for img_i in range(len(reps)):
                x = sims[top_i][img_i]
                x_test.append(x)
            res[top_i] = x_test
        
        self.testset = res
    
    def take_existing_imgs(self, imgs, reps):
        return [img for img in imgs if img in reps]

    def __build_dataset__(self):
        assert opt_xgb in self.opts
        assert opt_validation_size_positives in self.opts[opt_xgb]
        assert opt_validation_size_negatives in self.opts[opt_xgb]
        assert opt_seed in self.opts[opt_xgb]
        assert opt_cut_negatives in self.opts[opt_xgb]

        x1_train_tops, x1_valid_tops = train_test_split_gt(self.ctx.gt_dict(), self.opts[opt_xgb][opt_validation_size_positives], shuffle=True, seed=self.opts[opt_xgb][opt_seed])

        sims = self.train_model.sims
        reps = self.train_model.repr_imgs
        trn = {}
        val = {}

        for i in tqdm(range(len(self.train_model.ctx.tops()))):
            data_valid_x = []
            data_valid_y = []
            data_valid_imgs = []

            data_train_x = []
            data_train_y = []
            data_train_imgs = []

            top_id = self.train_model.ctx.tops()[i][top_idi]
            x1_valid = self.take_existing_imgs(x1_valid_tops[top_id], reps)
            x1_train = self.take_existing_imgs(x1_train_tops[top_id], reps)

            irrelevant_imgs_top = [x for x in reps if (x not in x1_valid) and (x not in x1_train)]
            
            negatives_count = len(irrelevant_imgs_top)
            
            if self.opts[opt_xgb][opt_cut_negatives] > 0:
                cut_negatives_count = ceil(len(irrelevant_imgs_top) * self.opts[opt_xgb][opt_cut_negatives])
                negatives_count = negatives_count - cut_negatives_count
                print("Ignoring {n} negatives for top {t}.".format(n=str(cut_negatives_count), t=str(top_id)))
        
            irrelevant_imgs_top = random.sample(irrelevant_imgs_top, negatives_count)
            x0_train, x0_valid = train_test_split_list(irrelevant_imgs_top, self.opts[opt_xgb][opt_validation_size_negatives], shuffle=True, seed=self.opts[opt_xgb][opt_seed])

            for img in x0_valid + x1_valid:
                similarity_ind = reps.index(img)
                x = sims[i][similarity_ind]
                data_valid_x.append(x)

            data_valid_y.extend([0] * len(x0_valid))
            data_valid_y.extend([1] * len(x1_valid))

            data_valid_imgs.extend(x0_valid)
            data_valid_imgs.extend(x1_valid)

            for img in x0_train + x1_train:
                similarity_ind = reps.index(img)
                x = sims[i][similarity_ind]
                data_train_x.append(x)

            data_train_y.extend([0] * len(x0_train))
            data_train_y.extend([1] * len(x1_train))

            data_train_imgs.extend(x0_train)
            data_train_imgs.extend(x1_train)

            trn[i] = (data_train_imgs, data_train_x, data_train_y)
            val[i] = (data_valid_imgs, data_valid_x, data_valid_y)

        self.trainset = trn
        self.validset = val

    def fit(self):
        if self.opts[opt_xgb][opt_predict_test]:
            self.__build_testset__()
        self.__build_dataset__()
