import numpy as np
from tqdm import tqdm

from src.io.ReadingContext import ReadingContext
from src.query_translation.get_weights import get_weight
from src.query_translation.get_clusters import get_cluster_indicies
from src.models.pooling.model_opts_to_token_clustering_opts import convert_to_token_clustering_opts
from src.models.pooling.Model_opts import *

class ClusterPrediction:
    def __init__(self, opts: dict, ctx: ReadingContext, data: list, vecs: list, repr_imgs: list, query_tokens: list):
        self.opts = opts
        self.ctx = ctx
        self.data = data
        self.vecs = vecs
        self.repr_imgs = repr_imgs
        self.query_tokens = query_tokens
    
    def fit(self):
        self.clusters = { }
        self.token_weights = { }
        token_clustering_opts = convert_to_token_clustering_opts(self.opts)
        for top_i, top in tqdm(enumerate(self.ctx.tops()), total=len(self.ctx.tops())):
            query_tokens = self.query_tokens[top_i]
            self.clusters[top_i] = get_cluster_indicies(query_tokens, token_clustering_opts, self.ctx.emb())
            if self.opts[opt_model][opt_comp_use_weights]:
                weigths_matrix = np.matrix([get_weight(query_tokens, data.tokenized_labels, self.ctx.emb()) for data in self.data])
                self.token_weights[top_i] = weigths_matrix
        
        for data in self.data:
            data_name = data.get_name()
            assert data_name in self.opts[opt_data]
            w = self.opts[opt_data][data_name][opt_weight]
            # weights must be between in range [0, 1]
            assert 0 <= w <= 1

        self.data_weights = np.array([self.opts[opt_data][d.get_name()][opt_weight] for d in self.data])

    def predict(self):
        img_ids_to_sims = { }

        for top_i, top in tqdm(enumerate(self.ctx.tops()), total=len(self.ctx.tops())):
            img_ids_to_sims[top_i] = { }
            
            if self.opts[opt_model][opt_comp_use_weights]:
                top_weights = self.token_weights[top_i]
            
            top_cluster_indicies = self.clusters[top_i]

            for img_i, img_repr in enumerate(self.repr_imgs):
                similarity_matrix = np.matrix([vec.similarities[top_i][img_i] for vec_i, vec in enumerate(self.vecs)])
                
                # multiply each row of similarity_matrix with the data_weight
                similarity_matrix = np.multiply(similarity_matrix.T, self.data_weights).T

                if self.opts[opt_model][opt_comp_use_weights]:
                    similarity_matrix = np.multiply(similarity_matrix, top_weights)

                cluster_sims = np.zeros(len(top_cluster_indicies))

                for cluster_indicies_i, cluster_indicies in enumerate(top_cluster_indicies):
                    cluster_matrix = similarity_matrix[:, cluster_indicies]
                    
                    if self.opts[opt_model][opt_comp_method] == comp_method_mean:
                        sim = cluster_matrix.mean()
                    elif self.opts[opt_model][opt_comp_method] == comp_method_datamax:
                        sim = cluster_matrix.max(0).mean()
                    elif self.opts[opt_model][opt_comp_method] == comp_method_tokenmax:
                        sim = cluster_matrix.max(1).mean()
                    else: assert False
                    
                    cluster_sims[cluster_indicies_i] = sim
                
                if self.opts[opt_tokenclustering][opt_tokenclustering_comp_method] == comp_method_clusters_mean:
                    clusters_sim = cluster_sims.mean()
                elif self.opts[opt_tokenclustering][opt_tokenclustering_comp_method] == comp_method_clusters_max:
                    clusters_sim = cluster_sims.max()
                else: assert False
                    
                if not (0 <= sim <= 1):
                    print(sim)
                    assert False
                    
                img_ids_to_sims[top_i][img_repr] = clusters_sim

        return img_ids_to_sims