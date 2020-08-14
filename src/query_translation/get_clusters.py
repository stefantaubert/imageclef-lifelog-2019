#from src.io.ReadingContext import ReadingContext
#from embeddings.w2v import __get_similarity_words__
#import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc

from gensim.models.keyedvectors import Word2VecKeyedVectors

from src.query_translation.get_clusters_opts import *

def flattening_clusters(cluster, threshold: float, criterion: str) -> list:
    fcluster = shc.fcluster(cluster, t=threshold, criterion=criterion)
    return fcluster

def build_clusters(emb: Word2VecKeyedVectors, tokens: list, method: str, metric: str):
    word_vectors = [emb[t] for t in tokens]
    cluster = shc.linkage(word_vectors, method=method, metric=metric)
    return cluster

def fcluster_to_dict(fcluster, tokens):
    res = {}
    for i, cluster_id in enumerate(fcluster):
        if cluster_id in res.keys():
            res[cluster_id].append(tokens[i])
        else:
            res[cluster_id] = [tokens[i]]

def fclusters_to_indicies(fcluster: list) -> list:
    cluster_ids = {x: [] for x in set(fcluster)}
    for i, c in enumerate(fcluster):
        cluster_ids[c].append(i)
    return list(cluster_ids.values())

def get_cluster_indicies(tokens: list, opts: dict, emb: Word2VecKeyedVectors) -> list:
    if len(tokens) == 1:
        return [[0]]
    else:
        cluster = build_clusters(emb, tokens, opts[opt_linkage_method], opts[opt_linkage_metric])
        fcluster = flattening_clusters(cluster, opts[opt_fcluster_threshold], opts[opt_fcluster_criterion])
        assert len(fcluster) > 1
        return fclusters_to_indicies(fcluster)

# def plt_clusters(cluster):
#     plt.figure(figsize=(10, 7))  
#     plt.title("Customer Dendograms")
#     dend = shc.dendrogram(cluster)
#     plt.show()

# def print_inter_word_similarities(ctx: ReadingContext, words: list):
#     for word in words:
#         other = [w for w in words if w != word]
#         sims = 1 - __get_similarity_words__(ctx.emb(), [word], other)
#         print(sims)

# def main(ctx: ReadingContext, words: list):
#     if len(words) == 1:
#         return words
#     else:
#         cluster = build_clusters(ctx, words)
#         plt_clusters(cluster)
#         fcluster = flattening_clusters(cluster, ctx)
#         d = fcluster_to_dict(fcluster, words)
#         print(d)


# if __name__ == "__main__":
#     from src.query_translation.get_query_tokens import get_query_tokens_from_topic
#     from src.query_translation.get_label_tokens import get_label_tokens
#     from src.globals import ds_dev
#     from src.globals import ds_test
#     from src.globals import top_desc
#     from src.models.pooling.Model_opts import *
#     

#     dev_opts = {
#         opt_general: {
#             opt_usr: 1,
#             
#             opt_ds: ds_dev,
#         },
        
#         opt_model: {
#             opt_query_src: query_src_title,
#         }
#     }

#     test_opts = {
#          opt_general: {
#             opt_usr: 1,
#             
#             opt_ds: ds_test,
#         },
        
#         opt_model: {
#             opt_query_src: query_src_title,
#         }
#     }

#     dev_ctx = Context(dev_opts)
#     test_ctx = Context(test_opts)
#     all_tops = []
#     for top in dev_ctx.tops(): all_tops.append(top)
#     for top in test_ctx.tops(): all_tops.append(top)

#     for top in all_tops:
#         query_tokens = get_query_tokens_from_topic(top, dev_ctx)
#         clusters = get_clusters(query_tokens, dev_ctx)
#         print(clusters)
#         #print_inter_word_similarities(dev_ctx, query_tokens)