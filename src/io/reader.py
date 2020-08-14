import gensim
import pickle
import pandas as pd
import xml.etree.ElementTree as ET

from src.io.paths import get_path_vec_6B_50d
from src.io.paths import get_path_vec_840B_300d
from src.io.paths import get_path_con
from src.io.paths import get_path_cat
from src.io.paths import get_path_att
from src.io.paths import get_path_vc
from src.io.paths import get_path_mbt
from src.io.paths import get_path_labels
from src.io.paths import get_path_tops
from src.io.paths import get_path_tops_prep
from src.io.paths import get_path_clusters
from src.io.paths import get_path_gt
from src.io.paths import get_dir_auto
from src.io.paths import get_dir_pers
from src.io.paths import get_dir_img
from src.io.paths import get_path_yolo
from src.io.paths import get_path_yolo_9k
from src.io.paths import get_path_yolo_openimgs
from src.io.paths import get_path_yolo_imgnet
from src.io.paths import get_path_detectron
from src.globals import top_id
from src.globals import top_idi
from src.globals import top_type
from src.globals import top_usr
from src.globals import top_title
from src.globals import top_desc
from src.globals import top_narrative
from tools.cell_array_converter import convert_mat

def read_csv(path):
    data = pd.read_csv(path, sep=',', encoding = "ISO-8859-1", low_memory=False, error_bad_lines=True)
    return data.fillna('')

def read_xml(path) -> list:
    result = []
    tree = ET.parse(path)
    root = tree.getroot()
    for topic in root:
        tmp = {}
        tmp[top_id] = topic[0].text.strip()
        tmp[top_idi] = int(tmp[top_id].lstrip("0"))
        tmp[top_type] = topic[1].text.strip()
        tmp[top_usr] = topic[2].text.strip()
        tmp[top_title] = topic[3].text.replace(u'\u200b ', '').strip() #???
        tmp[top_desc] = topic[4].text.strip()
        tmp[top_narrative] = topic[5].text.strip()
        result.append(tmp)
    
    return result

def read_vec(big=False):
    if big:
        path = get_path_vec_840B_300d()
    else:
        path = get_path_vec_6B_50d()
    
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
    return model

def read_labels():
    path = get_path_labels()
    data = pickle.load(open(path, "rb"))
    return data

def read_att():
    path = get_path_att()
    data = list(sorted(set(convert_mat(path, "attributes"))))
    return data

def read_con():
    path = get_path_con()
    f = open(path, "r")
    data = []

    for row in f:
        con = row.split(': ')[1].rstrip('\n').replace(' ', '_')
        data.append(con)

    data = set(sorted(data))
    return data

def read_cat():
    path = get_path_cat()
    f = open(path, "r")
    data = []
    
    for row in f:
        con = row[3:].split()[0]
        data.append(con)

    data = set(sorted(data))
    return data

def read_tops_prep():
    path = get_path_tops_prep()
    data = read_csv(path)
    return data

def read_detectron(usr=1, vers=None):
    path = get_path_detectron(usr=usr)
    data = read_csv(path)
    return data

def read_yolo_imgnet(usr=1, vers=None):
    path = get_path_yolo_imgnet(usr=usr)
    data = read_csv(path)
    return data

def read_oi(usr=1, vers=None):
    path = get_path_yolo_openimgs(usr=usr)
    data = read_csv(path)
    return data

def read_9k(usr=1, vers=None):
    path = get_path_yolo_9k(usr=usr)
    data = read_csv(path)
    return data

def read_yolo(usr=1, vers=None):
    path = get_path_yolo(usr=usr)
    data = read_csv(path)
    return data

def read_vc(usr=1, vers=None):
    path = get_path_vc(usr=usr, vers=vers)
    data = read_csv(path)
    return data

def read_mbt(usr=1, vers=None):
    path = get_path_mbt(usr=usr, vers=vers)
    data = read_csv(path)
    return data

def read_tops(ds=1) -> list:
    path = get_path_tops(ds=ds)
    xml = read_xml(path)
    return xml

def read_clusters():
    path = get_path_clusters()
    data = read_csv(path)
    return data

def read_gt():
    path = get_path_gt()
    data = read_csv(path)
    return data