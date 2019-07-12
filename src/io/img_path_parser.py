import pickle

from pathlib import Path

from src.io.paths import get_path_from_rel_img_path
from src.io.reader import read_vc
from src.io.data_dir_config import root

def __get_paths_csv__(vc, usr=1, root=root):
    ids = vc["image_id"]
    rel_paths = vc["image_path"]
    paths = [get_path_from_rel_img_path(p, usr, root) for p in rel_paths]
    rows_c = len(ids)
    #res = [(ids[i], paths[i]) for i in range(rows_c)]
    res = dict((ids[i], paths[i]) for i in range(rows_c))
    return res

def get_paths(usr=1, root=root):
    vc = read_vc(usr)
    paths = __get_paths_csv__(vc, usr, root)
    return paths

def get_paths_cached(usr=1, root=root):
    cache_path = "/tmp/paths_dict_{usr}.pkl".format(usr=str(usr))
    is_in_cache = Path(cache_path).exists()
    if is_in_cache:
        with open(cache_path, 'rb') as f:
            paths = pickle.load(f)
    else:
        vc = read_vc(usr)
        paths = __get_paths_csv__(vc, usr, root)
        with open(cache_path, 'wb') as f:
            pickle.dump(paths, f)
    return paths
