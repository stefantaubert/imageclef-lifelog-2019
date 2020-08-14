from src.io.ReadingContext import ReadingContext
from src.models.pooling.Model import Model as PoolingModel
from src.models.xgb.Model import Model
from src.models.xgb.Model_opts import *
from src.models.pooling.Model_opts import *

from src.globals import ds_dev
from src.globals import ds_test
import copy

class XGBModel():
    def __init__(self, ctx: ReadingContext):
        self.ctx = ctx

    def fit(self):
        dev_opts = copy.deepcopy(self.ctx.opts)
        dev_opts[opt_general][opt_ds] = ds_dev
        m_train = PoolingModel(dev_opts)
        m_train.fit()

        test_opts = copy.deepcopy(self.ctx.opts)
        test_opts[opt_general][opt_ds] = ds_test
        m_test = PoolingModel(test_opts)
        
        if self.ctx.opts[opt_xgb][opt_predict_test]:
            m_test.fit()

        xgb_ctx = ReadingContext(copy.deepcopy(self.ctx.opts))
        self.xgb_model = Model(xgb_ctx, m_train, m_test)
        self.xgb_model.fit()
    
    def predict(self):
        self.xgb_model.train()
        self.xgb_model.predict()

    def run(self):
        self.fit()
        self.predict()
