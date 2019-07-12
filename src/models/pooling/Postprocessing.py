from src.io.ReadingContext import ReadingContext
from sklearn.pipeline import Pipeline
from src.transformers.MinMaxPrinterTransformer import MinMaxPrinterTransformer
from src.transformers.OrderingTransformer import OrderingTransformer
from src.transformers.RearrangeTransformer import RearrangeTransformer
from src.globals import top_idi
from src.globals import top_type
from src.models.pooling.Model_opts import *

class Postprocessing():
    def __init__(self, opts: dict, sims, tops):
        self.opts = opts
        self.sims = sims
        self.tops = tops
    
    def fit(self):
        assert opt_model in self.opts
        assert opt_subm_imgs_per_day in self.opts[opt_model]

        self.sort_pipe = Pipeline([
            #("print_min_max", MinMaxPrinterTransformer()),
            ("sort", OrderingTransformer(reverse=True)),
        ])

        self.recall_pipe = Pipeline([
            ("rearrange", RearrangeTransformer(self.opts[opt_model][opt_subm_imgs_per_day])),
        ])
    
    def process(self):
        optimize_only_recall = self.opts[opt_model][opt_subm_imgs_per_day_only_on_recall]

        subm = { }
        
        for i, top in enumerate(self.tops):
            is_recall_type = top[top_type] == "recall"
            top_sims = self.sims[i]
            top_subm = self.sort_pipe.transform(top_sims)
            if (optimize_only_recall and is_recall_type) or (not optimize_only_recall):
                top_subm = self.recall_pipe.transform(top_subm)

            top_id = self.tops[i][top_idi]
            subm[top_id] = top_subm

        return subm