import pandas as pd
import pickle

from pathlib import Path

from src.io.paths import get_dir_cache
from src.io.reader import read_vc
from src.io.reader import read_mbt
from src.io.reader import read_labels
from src.io.reader import read_tops
from src.io.reader import read_vec
from src.io.reader import read_oi
from src.io.reader import read_detectron
from src.io.reader import read_yolo_imgnet
from src.io.reader import read_yolo
from src.io.reader import read_gt
from src.common.gt_converter import gt_to_dict
from src.common.helper import create_parent_dir
from gensim.models.keyedvectors import Word2VecKeyedVectors

class ReadingContext():
    def __init__(self, usr: int = 1, ds: int = 1):
        self.usr = usr
        self.ds = ds

    def gt(self) -> pd.DataFrame: 
        try:
            return self.__ground_truth__
        except:
            self.__ground_truth__ = read_gt()
            return self.__ground_truth__
    
    def gt_dict(self) -> dict: 
        try:
            return self.__ground_truth_dict__
        except:
            self.__ground_truth_dict__ = gt_to_dict(self.gt())
            return self.__ground_truth_dict__
    
    def vc(self) -> pd.DataFrame:
        try:
            return self.__vc__
        except:
            self.__vc__ = read_vc(usr=self.usr)
            return self.__vc__
    
    def vc_dict(self) -> dict:
        try:
            return self.__vc_dict__
        except:
            self.__vc_dict__ = self.vc().set_index("image_id").to_dict('index')
            return self.__vc_dict__
    
    def mbt(self) -> pd.DataFrame:
        try:
            return self.__mbt__
        except:
            self.__mbt__ = read_mbt(usr=self.usr)
            return self.__mbt__
    
    def mbt_dict(self) -> dict:
        try:
            return self.__mbt_dict__
        except:
            self.__mbt_dict__ = self.mbt().set_index("minute_ID").to_dict('index')
            return self.__mbt_dict__
    
    def imagenet(self) -> pd.DataFrame:
        try:
            return self.__imagenet__
        except:
            self.__imagenet__ = read_yolo_imgnet(usr=self.usr)
            return self.__imagenet__
    
    def detectron(self) -> pd.DataFrame:
        try:
            return self.__detectron__
        except:
            self.__detectron__ = read_detectron(usr=self.usr)
            return self.__detectron__
    
    def yolo(self) -> pd.DataFrame:
        try:
            return self.__yolo__
        except:
            self.__yolo__ = read_yolo(usr=self.usr)
            return self.__yolo__
    
    def openimages(self) -> pd.DataFrame:
        try:
            return self.__openimages__
        except:
            self.__openimages__ = read_oi(usr=self.usr)
            return self.__openimages__
    
    def labels(self) -> dict:
        try:
            return self.__labels__
        except:
            self.__labels__ = read_labels()
            return self.__labels__
    
    def tops(self) -> list:
        try:
            return self.__topics__
        except:
            self.__topics__ = read_tops(ds=self.ds)
            return self.__topics__
    
    def emb(self) -> Word2VecKeyedVectors:
        try:
            return self.__embeddings__
        except:
            print("Load embeddings...")
            cache = get_dir_cache(usr=self.usr)
            cached_emb = cache + "embeddings.pkl"
        
            if Path(cached_emb).exists():
                self.__embeddings__ = pickle.load(open(cached_emb, "rb"))
                return self.__embeddings__
            else:
                self.__embeddings__ = read_vec(big=True) #TODO remove parameter
                pickle.dump(self.__embeddings__, open(cached_emb, "wb"))
                return self.__embeddings__

    def vocab(self) -> set:
        try:
            return self.__vocab__
        except:
            self.__vocab__ = set(self.emb().vocab.keys())
            return self.__vocab__