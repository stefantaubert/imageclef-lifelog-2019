import pickle

from src.io.paths import get_path_labels

def write_labels(labels):
    path = get_path_labels()
    data = pickle.dump(labels, open(path, "wb"))
    return data
