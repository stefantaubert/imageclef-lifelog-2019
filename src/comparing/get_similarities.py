import numpy as np

from tqdm import tqdm
from joblib import Parallel
from joblib import delayed
from operator import itemgetter
from multiprocessing import cpu_count

from src.word2vec.normalize_cosine_similarity import normalize_cosine_similarity
from src.comparing.get_similarities_opts import *

def take_non_zero_entries(v1, v2, idfs):
    assert len(v1) == len(v2)
    assert len(v1) == len(idfs)
    relevant_is = [i for i in range(len(v1)) if v1[i] != 0 and v2[i] != 0]
    v1_res = np.array([v1[i] for i in relevant_is])
    v2_res = np.array([v2[i] for i in relevant_is])
    idfs_res = np.array([idfs[i] for i in relevant_is])
    return (v1_res, v2_res, idfs_res)

def __get_similarity_idf_top_bottom__(vec1, vec2, idfs, idf_t: float, m: int, p: int, ceiling: bool):
    if ceiling: vec1 = np.ceil(vec1)
    vec2 = np.array([normalize_cosine_similarity(v) for v in vec2])
    diff = np.absolute(vec1 - vec2)
    w = np.array([1 / idfs[i] if diff[i] <= idf_t else idfs[i] for i in range(len(diff))])
    top = (m * diff) ** p
    top = np.multiply(top, w)
    bottom = (m ** p) * idfs
    return (top, bottom)

def __get_similarity_idf__(vec1, vec2, idfs, idf_t: float, m: int, p: int, ceiling: bool):
    res = []
    for v2 in vec2:
        v1, v2, idf = take_non_zero_entries(vec1, v2, idfs)
        if len(v1) == 0:
            res.append(0)
            continue

        top, bottom = __get_similarity_idf_top_bottom__(v1, v2, idf, idf_t, m, p, ceiling)
        top_norm = np.linalg.norm(top)
        bottom_norm = np.linalg.norm(bottom)
        sim = 1 - (top_norm / bottom_norm)
        res.append(sim)
    return res

def __get_similarity_noidf__(vec1, vec2, m: int, p: int, ceiling: bool):
    idfs = [1] * len(vec1)
    return __get_similarity_idf__(vec1, vec2, idfs, 0, m, p, ceiling)

def __get_similarities_imgs__(i: int, img_vecs, que_vecs, idfs, opts: dict):
    res = [0] * len(img_vecs)
    que_vec = que_vecs[i]
    for img_vec_i, img_vec in enumerate(img_vecs):
        #relevant_img_vec = [x if x >= opts["score_threshold"] else 0 for x in img_vecs[j]]
        if opts["use_idf"]:
            res[img_vec_i] = __get_similarity_idf__(img_vec, que_vec, idfs, opts[opt_idf_boosting_threshold], opts[opt_intensify_factor_m], opts[opt_intensify_factor_p], opts[opt_ceiling])
        else:
            res[img_vec_i] = __get_similarity_noidf__(img_vec, que_vec, opts[opt_intensify_factor_m], opts[opt_intensify_factor_p], opts[opt_ceiling])
                
    return (i, res)

def __get_similarities_sequential__(img_vecs, que_vecs, idfs, opts: dict):
    similarities = []
    for i in tqdm(range(len(que_vecs))):
        _, sims = __get_similarities_imgs__(i, img_vecs, que_vecs, idfs, opts)
        similarities.append(sims)
    return similarities

def __get_similarities_parallel__(img_vecs, que_vecs, idfs, opts: dict):
    similarities = []
    result = Parallel(n_jobs=cpu_count())(delayed(__get_similarities_imgs__)(i, img_vecs, que_vecs, idfs, opts) for i in tqdm(range(len(que_vecs))))
    result = sorted(result,key=itemgetter(0))
    #print([x for x, _ in result])
    similarities = [s for _, s in result]
    return similarities

def get_similarities(img_vecs, que_vecs, idfs, opts: dict):
    assert opt_use_idf in opts
    assert opt_idf_boosting_threshold in opts
    assert opt_intensify_factor_m in opts
    assert opt_intensify_factor_p in opts
    assert opt_ceiling in opts
    assert opt_multiprocessing in opts
    o = dict(opts)
    if o[opt_multiprocessing]: 
        return __get_similarities_parallel__(img_vecs, que_vecs, idfs, o)
    else: 
        return __get_similarities_sequential__(img_vecs, que_vecs, idfs, o)
