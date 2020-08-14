import numpy as np
from tqdm import tqdm

from src.io.ReadingContext import ReadingContext
from src.query_translation.get_query_tokens import get_query_tokens_from_topic
from src.query_translation.get_weights import get_weight
from src.models.pooling.Model_opts import *

class XGBTokenPrediction:
    def __init__(self, ctx: ReadingContext):
        self.ctx = ctx
    
    def fit(self):
        assert opt_model in self.ctx.opts
        assert opt_data in self.ctx.opts
        assert opt_comp_use_weights in self.ctx.opts[opt_model]

        if self.ctx.opts[opt_model][opt_comp_use_weights]:
            self.token_weights = { }
            for top_i, top in tqdm(enumerate(self.ctx.tops()), total=len(self.ctx.tops())):
                query_tokens = get_query_tokens_from_topic(top, self.ctx)
                weigths_matrix = np.matrix([get_weight(query_tokens, data, self.ctx) for data in self.ctx.mem[mem_data]])
                self.token_weights[top_i] = weigths_matrix

    def predict(self):
        count_tops = len(self.ctx.tops())
        count_imgs = len(self.ctx.mem[mem_preprocessing].representatives())
        count_data = len(self.ctx.mem[mem_vecs])
        img_ids_to_sims = np.empty((count_tops, count_imgs, count_data))

        for top_i, top in tqdm(enumerate(self.ctx.tops()), total=len(self.ctx.tops())):
            
            if self.ctx.opts[opt_model][opt_comp_use_weights]:
                top_weights = self.token_weights[top_i]

            for img_i, img_repr in enumerate(self.ctx.mem[mem_preprocessing].representatives()):
                similarity_matrix = np.matrix([vec.similarities[top_i][img_i] for _, vec in enumerate(self.ctx.mem[mem_vecs])])
                
                if self.ctx.opts[opt_model][opt_comp_use_weights]:
                    similarity_matrix = np.multiply(similarity_matrix, top_weights)

                if self.ctx.opts[opt_model][opt_comp_method] == comp_method_mean:
                    sim = similarity_matrix.mean(1)
                elif self.ctx.opts[opt_model][opt_comp_method] == comp_method_tokenmax:
                    sim = similarity_matrix.max(1)
                else: assert False
                
                for sim_i, vec_sim in enumerate(sim):
                    img_ids_to_sims[top_i][img_i][sim_i] = vec_sim

        self.ctx.mem[mem_sims] = img_ids_to_sims