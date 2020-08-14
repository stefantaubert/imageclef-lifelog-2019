# https://docs.python.org/3/library/time.html#time.strftime

from datetime import datetime
from math import isnan
from re import sub
from os import makedirs
from os import listdir
from os import pardir
from os.path import exists
from os.path import isfile
from os.path import join
from os.path import abspath
from os.path import basename
from shutil import rmtree
from shutil import copyfile

from src.io.paths import get_dir_cache

def remove_bad_path_chars(path):
    return sub('[^\w\-_\. ]', '_', path)

def extract_datetime(utc_time):
    return datetime.strptime(utc_time, "%Y%m%d_%H%M_UTC")

def extract_date(utc_time):
    return datetime.date(extract_datetime(utc_time))

def extract_time(utc_time):
    return datetime.time(extract_datetime(utc_time))

def delete_minute_in_time(time):
    time_str = time.strftime("%Y%m%d%H00")
    return datetime.time(datetime.strptime(time_str, "%Y%m%d%H%M"))

def round_to_full_minute(time):
    time_str = time.strftime("%Y%m%d%H%M")
    time_str = time_str[:-1] + "0"
    return datetime.time(datetime.strptime(time_str, "%Y%m%d%H%M"))

'''Extracts datetime from strings like: B00008061_21I6X0_20180506_105818E.JPG'''
def extract_autographer_datetime(image_name):
    image_name = image_name[-20:]
    return datetime.strptime(image_name, "%Y%m%d_%H%M%SE.JPG")

'''Extracts datetime from strings like: 2018-05-10 08.06.48.jpg'''
def extract_personal_datetime(image_name):
    return datetime.strptime(image_name, "%Y-%m-%d %H.%M.%S.jpg")

def time_to_datetime(time):
    time_str = time.strftime("19700101%H%M")
    return datetime.strptime(time_str, "%Y%m%d%H%M")

def check_is_nan(value, is_float):
    if is_float:
        return isnan(value)
    else:
        # strip() weil es schon vorkam dass es " NULL" war
        val = str(value).strip()
        return val == 'None' or val == 'nan' or val == 'NULL'

def create_dir_if_not_exists(path):
    if not exists(path):
        makedirs(path)

def get_empty_dir(dir):
    if not exists(dir):
        makedirs(dir)
    else:
        rmtree(dir)
        makedirs(dir)

def get_filepaths(dir):
    onlyfiles = [dir + f for f in listdir(dir) if isfile(join(dir, f)) and (f[0] != ".")]
    onlyfiles = sorted(onlyfiles)
    return onlyfiles

def clear_cache(usr=1):
    cache = get_dir_cache(usr=usr)
    get_empty_dir(cache)
    
def copy_to(origin_filenames, destination_dir):
    assert len(destination_dir) > 0
    if destination_dir[-1] != "/":
        destination_dir = destination_dir + "/"

    get_empty_dir(destination_dir)

    for file in origin_filenames:
        file_name = basename(file)
        dest_file = "{dest_dir}{filename}".format(dest_dir=destination_dir, filename=file_name)
        copyfile(file, dest_file)

def get_parent_dir(filename):
    parent_dir = abspath(join(filename, pardir))
    return parent_dir

def create_parent_dir(filename):
    parent_dir = get_parent_dir(filename)
    create_dir_if_not_exists(parent_dir)

def copy(origin_filenames, destination_filenames):
    assert len(origin_filenames) == len(destination_filenames)
    
    for i in range(len(origin_filenames)):
        create_parent_dir(destination_filenames[i])
        copyfile(origin_filenames[i], destination_filenames[i])