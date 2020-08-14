from src.io.data_dir_config import root
import os

def get_path_from_rel_img_path(rel_path, usr=1, root=root):
    dir_pers = "u{usr}_photos".format(usr=usr)
    is_cam = rel_path.startswith(dir_pers)
    
    return get_path_from_rel_path(rel_path, is_cam, usr=usr, root=root)

def get_path_from_rel_path(rel_path, is_cam, usr=1, root=root):
    if is_cam:
        img_dir = get_dir_img(usr=usr, root=root)
        img_fullpath = "{img_dir}{img_path}".format(img_dir=img_dir, img_path=rel_path)
    else:
        auto_dir = get_dir_auto(usr=usr, root=root)
        img_fullpath = "{img_dir}{img_path}".format(img_dir=auto_dir, img_path=rel_path)
    
    return img_fullpath

def get_path_labels(root=root):
    return "{root}res/labels.pkl".format(root=root)

def get_path_darknet(root=root):
    return "{root}res/darknet/".format(root=root)

def get_path_vec_840B_300d(root=root):
    return "{root}res/glove/glove.840B.300d.txt.word2vec".format(root=root)

def get_path_vec_6B_50d(root=root):
    return "{root}res/glove/glove.6B.50d.txt.word2vec".format(root=root)

def get_path_vec(root=root):
    return "{root}res/glove.6B/glove.6B.50d.w2v.txt".format(root=root)

def get_path_con(root=root):
    return "{root}res/concepts.txt".format(root=root)

def get_path_cat(root=root):
    return "{root}res/categories.txt".format(root=root)

def get_path_att(root=root):
    return "{root}res/attributes.mat".format(root=root)

def get_path_vc(root=root, usr=1, vers=None):
    append_vers = "_{vers}".format(vers=str(vers)) if vers != None else ""
    return "{root}data/visual_concepts/u{usr}_categories_attr_concepts{vers}.csv".format(root=root, usr=usr, vers=append_vers)

def get_path_detectron(root=root, usr=1):
    return "{root}data/visual_concepts/u{usr}_detectron.csv".format(root=root, usr=usr)

def get_path_yolo_9k(root=root, usr=1):
    return "{root}data/visual_concepts/u{usr}_yolo_9k.csv".format(root=root, usr=usr)

def get_path_yolo_imgnet(root=root, usr=1):
    return "{root}data/visual_concepts/u{usr}_yolo_imagenet1k.csv".format(root=root, usr=usr)

def get_path_yolo(root=root, usr=1):
    return "{root}data/visual_concepts/u{usr}_yolo.csv".format(root=root, usr=usr)

def get_path_yolo_openimgs(root=root, usr=1):
    return "{root}data/visual_concepts/u{usr}_yolo_openimages.csv".format(root=root, usr=usr)

def get_path_mbt(root=root, usr=1, vers=None):
    append_vers = "_{vers}".format(vers=str(vers)) if vers != None else ""
    return "{root}data/minute_based_table/u{usr}{vers}.csv".format(root=root, usr=usr, vers=append_vers)

def get_path_clusters(root=root):
    return "{root}dev/lmrt/clusters.csv".format(root=root)

def get_path_gt(root=root):
    return "{root}dev/lmrt/LMRT_gt.csv".format(root=root)

def get_path_tops_prep(root=root):
    return "{root}dev/lmrt/topics_prep.csv".format(root=root)

def get_path_tops(root=root, ds=1):
    subdir = "dev" if ds == 1 else "test"
    return "{root}{subdir}/lmrt/topics.xml".format(root=root, subdir=subdir)

def get_path_sgmts(root=root, usr=1):
    return "{root}res/u{usr}_segments.pkl".format(root=root, usr=str(usr))

def get_dir_img(root=root, usr=1):
    return "{root}data/u{usr}/".format(root=root, usr=str(usr))

def get_dir_cache(root=root, usr=1):
    return "{root}cache/u{usr}/".format(root=root, usr=str(usr))

def get_dir_subm(root=root):
    return "{root}submissions/".format(root=root)

def get_dir_auto(root=root, usr=1):
    img = get_dir_img(root, usr)
    return "{img}Autographer/".format(img=img)

def get_dir_pers(root=root, usr=1):
    img = get_dir_img(root, usr)
    return "{img}u{usr}_photos/".format(img=img, usr=str(usr))

def get_an_root(root=root):
    return "{root}analysis/".format(root=root)

def get_antype_root(root=root, t=1, usr=1):
    top_root = get_an_root(root)

    if t == 1:
        an_type = "data" 
    elif t == 2:
        an_type = "dev"
    elif t == 3:
        an_type = "test"
    elif t == 4:
        an_type = "model"
    else:
        raise AssertionError()
    
    return "{top_root}{an_type}/u{usr}/".format(top_root=top_root, an_type=an_type, usr=usr)

def mk_dir(name, t=1, usr=1, root=root):
    root = get_antype_root(root=root, t=t, usr=usr)
    path = "{root}{name}/".format(root=root, name=name)
    
    if not os.path.exists(path):
        os.makedirs(path)

    return path