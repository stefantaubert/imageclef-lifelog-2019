"""
Gets all tokens from a label.
returns only tokens which are in the embeddings.
"""
def get_label_tokens(labels, vocab: set):
    possible_labels = [[w] if w in vocab else w.split() for w in labels]    
    for i, _ in enumerate(possible_labels):
        possible_labels[i] = [w for w in possible_labels[i] if w in vocab]
    return possible_labels
