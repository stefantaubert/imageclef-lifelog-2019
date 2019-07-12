from gensim.models.keyedvectors import Word2VecKeyedVectors
#https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.Word2VecKeyedVectors

# for multiple words I build the average of the words
# ref: https://stackoverflow.com/questions/46889727/word2vec-what-is-best-add-concatenate-or-average-word-vectors

def get_similarities(query_token: str, tokenized_labels: list, emb: Word2VecKeyedVectors) -> list:
    sims = [__get_similarity_words__(emb, [query_token], label_tokens) for label_tokens in tokenized_labels]
    return sims

def __get_similarity_words__(embeddings: Word2VecKeyedVectors, words: list, other_words: list) -> float:
    if len(words) == 0 or len(other_words) == 0:
        return 0
        
    summed_avgs = 0
    for w in words:
        dist = []
        for o_w in other_words:
            sim = embeddings.similarity(w, o_w)
            dist.append(sim)
        avg = sum(dist) / len(other_words)
        summed_avgs += avg

    return summed_avgs / len(words)
