# Code for ImageCLEF 2019 Lifelog Challenge

Code for ImageCLEFlifelog 2019.

# Get it running


## Step 1 - Download dataset

### Prepare Folders

```bash
mkdir '/mnt/hdd/sttau/datasets/ImageCLEF 2019/Lifelog'
cd '/mnt/hdd/sttau/datasets/ImageCLEF 2019/Lifelog'
mkdir 'data'
mkdir 'dev/lmrt'
```

### Download dataset into the data dir and run following commands

```bash
cd 'data'
sudo apt install zip
zip -s- ImageCLEF2019lifelog_u1 -O u1
zip -s- ImageCLEF2019lifelog_u2 -O u2
unzip -P ImageCLEF2019lifelog u1
unzip -P ImageCLEF2019lifelog u2
unzip -P ImageCLEF2019lifelog_minutes
unzip -P ImageCLEF2019lifelog_visual
```

### Download devset into the dev dir and transform the data
TODO

- clusters.csv
- LMRT_gt.csv
- topics.xml

## Step 2 - Set development environment

```bash
mkdir '/mnt/hdd/sttau/code'
cd '/mnt/hdd/sttau/code'
git clone git@gitlab.hrz.tu-chemnitz.de:sttau--tu-chemnitz.de/lifelog-2019.git
pip install –r requirements.txt
nano src/data_dir_config.py
```

Now write the datapath from Step 1 into the file like so and save:

```py
root = "/mnt/hdd/sttau/datasets/ImageCLEF 2019/Lifelog"
```

On VS Code set env:

```json
"env": {"PYTHONPATH":"/mnt/hdd/sttau/code/lifelog-2019"}
```

You need Python 3.

## Step 3 - Getting Yolo

First install Cuda 10.1 if you want to run with GPU

```bash
mkdir '/mnt/hdd/sttau/datasets/ImageCLEF 2019/Lifelog/res'
git clone https://github.com/pjreddie/darknet
cd darknet
wget https://pjreddie.com/media/files/yolov3.weights
nano Makefile
```

set GPU=1 and save

```bash
sudo make
```

adjust file data/coco.data so that names = absolute location of coco.names

## Step 4 - Adjusting Files

Änderungen, die im Datenset vorgenommen wurden:

- line 12080: expected 50 fields, saw 51 (u2_20180517_0918)
- line 12081: expected 50 fields, saw 51 (u2_20180517_0919)
- line 12145: expected 50 fields, saw 51 (u2_20180517_1023)

Typo
2018_05_14/B00002730_21I6X0_20180514_145029E.jpg => .JPG

TODO:
Einträge säubern: The following 22 image_id's exist in the minute_based_table/u1.csv but aren't recorded in the u1_categories_attr_concepts.csv
Bilder annotieren und hinzufügen (optional): The following 27 images exist on the filesystem but are not recorded in the u1_categories_attr_concepts.csv

## Step 5 - Install XGBoost

For faster training install xgboost with gpu-support:
- on windows you find an intruction [here](http://www.picnet.com.au/blogs/guido/post/2016/09/22/xgboost-windows-x64-binaries-for-download/).
- on linux you find an instruction [here](https://github.com/dmlc/xgboost/blob/master/doc/build.md), td;tr:
    - if you have old version of cmake: deinstall old version and install new version from [here](https://cmake.org/download/) to e.g. `usr/local/` and create symlink `/usr/local/bin/cmake`
    - open terminal in e.g. `$root$/res`

    ```shell
    git clone --recursive https://github.com/dmlc/xgboost
    cd xgboost
    mkdir build
    cd build
    cmake .. -DUSE_CUDA=ON
    make -j4
    cd ..
    conda activate <env-name>
    pip install setuptools
    cd python-package; python setup.py install
    ```

Without gpu-support: `pip install xgboost`