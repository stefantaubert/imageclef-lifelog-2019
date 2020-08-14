from src.segmentation.CachableTransformerBase import CachableTransformerBase
import numpy as np
from tqdm import tqdm
import cv2 as cv
from imageio import imread

def get_hist(img):
    hist_full = cv.calcHist([img],[0],None,[256],[0,256])
    return hist_full

# get hist directly to hold ram usage low
def read_histo(path):
    img = imread(path)
    hist = get_hist(img)
    return hist

def get_histos(paths):
    res = []

    for path in tqdm(paths):
        hist = read_histo(path)
        res.append(hist)

    return np.array(res)

def flatten_hist(arr):
    nsamples, nx, ny = arr.shape
    d2_train_dataset = arr.reshape((nsamples,nx*ny))
    return d2_train_dataset

def get_histogram_vector(img_paths):
    histos = get_histos(img_paths)
    histos = flatten_hist(histos)
    return histos

class HistogramTransformer(CachableTransformerBase):
    """
    Calculates the histogram for the given images.
    Input: list of images on drive
    Output: list of histograms, which are a list of values themselfes
    """
    def __init__(self, day: int, usr: int, is_dirty: bool = False):
        return super().__init__(usr=usr, is_dirty=is_dirty, suffix=str(day))
        
    def before_transform(self, _):
        print("Extracting histograms...")
    
    def transform_core(self, day_img_paths: list):
        histos = get_histogram_vector(day_img_paths)
        return histos