from os.path import basename
from collections import OrderedDict

from analysis.paths import mk_dir
from src.common.helper import copy
from src.common.helper import get_empty_dir
from src.globals import an_type_test
from src.globals import an_type_dev
from src.globals import ds_test
from src.io.ReadingContext import ReadingContext
from src.io.img_path_parser import get_paths_cached

from src.models.pooling.Model_opts import opt_general
from src.models.pooling.Model_opts import opt_ds
from src.models.pooling.Model_opts import opt_usr

def show_in_fs_ctx(subm: dict, ctx: ReadingContext, model_name: str, top_n: int):
    return show_in_fs(subm, ctx.usr, ctx.ds, model_name, top_n)

def show_in_fs(subm, usr, ds, model_name: str, top_n: int):
    data_an_type = an_type_test if ds == ds_test else an_type_dev
    dest = mk_dir("submissions/{model}".format(model=model_name), usr=usr, t=data_an_type)
    get_empty_dir(dest)
    origin_paths = []
    dest_paths = []
    paths = get_paths_cached(usr)

    for top_id, preds in subm.items():
        assert isinstance(preds, OrderedDict)
        best_results = list(preds.keys())[:top_n]
        best_scores = list(preds.values())[:top_n]
        for i in range(top_n):
            img = best_results[i]
            origin_file = paths[img]
            dest_file = "{dir}{top_id}/{pos}-{img}-{score}.jpg".format(dir=dest, top_id=str(top_id), pos=str(i+1).zfill(2), img=img, score=str(best_scores[i]))
            origin_paths.append(origin_file)
            dest_paths.append(dest_file)

    copy(origin_paths, dest_paths)
    return dest
    
if __name__ == "__main__":
    subm = {
        1: OrderedDict({
            'u1_20180528_1816_i00': 1.0,
            'u1_20180508_1106_i00': 0.6,
        }),
        2: OrderedDict({
            'u1_20180528_1819_i00': 1.0,
            'u1_20180508_1106_i01': 0.8,
        }),
        3: OrderedDict({
            'u1_20180514_1117_i07': 0.46,
            'u1_20180508_1119_i02': 0.31,
        }),
    }

    show_in_fs(subm, 1, ds_test, "Test", 1)
