import scipy.io as sio
import numpy as np

def convert_mat(path, data_key):
    mat_contents = sio.loadmat(path)
    np_data = mat_contents[data_key]
    data = [x[0][0] for x in np_data]
    return data

