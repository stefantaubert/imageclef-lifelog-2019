import numpy as np

from src.io.ReadingContext import ReadingContext
from src.word2vec.get_similarities import get_similarities
from src.data.DataBase import DataBase

def get_weight(query_tokens: list, tokenized_labels: list, emb):
    highest_sims = np.zeros(len(query_tokens))
    has_tokenized_labels = len(tokenized_labels) > 0

    if has_tokenized_labels:
        for query_token_i, query_token in enumerate(query_tokens):
            sims = get_similarities(query_token, tokenized_labels, emb)
            highest_sim = max(sims)
            # sims can be negative
            highest_sims[query_token_i] = max(0, highest_sim)
    
    return highest_sims

# if __name__ == "__main__":
#     from src.query_translation.get_query_tokens import get_query_tokens
#     from src.globals import ds_dev
#     from src.globals import ds_test
#     from src.globals import top_desc

#     dev_opts = {
#         "usr": 1,
#         "large_embeddings": True,
#         "dataset": ds_dev,
#     }

#     test_opts = {
#         "usr": 1,
#         "large_embeddings": True,
#         "dataset": ds_test,
#     }

#     dev_ctx = Context(dev_opts)
#     test_ctx = Context(test_opts)
#     all_tops = []
#     for top in dev_ctx.tops(): all_tops.append(top)
#     for top in test_ctx.tops(): all_tops.append(top)

#     for top in all_tops:
#         query = top[top_desc]
#         query_tokens = get_query_tokens(query, dev_ctx.emb())
#         for query_token in query_tokens:
#             print("Most relevant to:", query_token)
#             print(get_weights(query_token, dev_ctx))