def get_dist(embeddings, sent1, sent2):
    words = sent1.split()
    other_words = sent2.split()
    other_words2 = remove_unknown_words(embeddings, other_words)
    if other_words2 != other_words:
        print("removed sth:", other_words, other_words2)
    other_words = other_words2
    summed_avgs = 0

    for w in words:
        if len(other_words) > 0:
            dist = embeddings.distances(w, other_words)
            avg = sum(dist) / len(other_words)
            summed_avgs += avg

    return summed_avgs / len(words)

def get_dist_words(embeddings, words, possible_words):
    #res = embeddings.similar_by_word(word)
    other_words = list(possible_words)
    prob = [get_dist(embeddings, words, other_word) for other_word in other_words]
    #prob = embeddings.distances(word, other_words)
    #s_prob, s_other_words = zip(*sorted(zip(prob, other_words)))
    result = { other_words[i]: prob[i] for i in range(len(prob)) }
    return result

def remove_unknown_words(embeddings, words):
    voc_words = []
    for w in words:
        if w in embeddings.vocab:
            voc_words.append(w)
    return voc_words