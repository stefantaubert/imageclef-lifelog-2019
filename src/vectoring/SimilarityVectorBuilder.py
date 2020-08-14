import numpy as np

from src.word2vec.get_similarities import get_similarities
from src.vectoring.VectorBuilderBase import VectorBuilderBase

class SimilarityVectorBuilder(VectorBuilderBase):
    def __init__(self, tokenized_labels: list, emb):
        self.emb = emb
        self.tokenized_labels = tokenized_labels
    
    def build_vector(self, query_tokens: list):
        vecs = []
        
        for query_token in query_tokens:
            sims = get_similarities(query_token, self.tokenized_labels, self.emb)
            sims = np.array(sims)
            vecs.append(sims)

        return vecs
