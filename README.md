# Automated Lifelog Moment Retrieval based on Image Segmentation and Similarity Scores

By [Stefan Taubert](https://stefantaubert.com/), [Stefan Kahl](http://medien.informatik.tu-chemnitz.de/skahl/about/), [Danny Kowerko](https://www.tu-chemnitz.de/informatik/mc/staff.php.en) and [Maximilian Eibl](https://www.tu-chemnitz.de/informatik/HomePages/Medieninformatik/team.php.en)

## Introduction

This is the sourcecode for our submissions to the ImageCLEFlifelog 2019 task.

Contact:  [Stefan Taubert](https://stefantaubert.com/), [Technische Universität Chemnitz](https://www.tu-chemnitz.de/index.html.en), [Media Informatics](https://www.tu-chemnitz.de/informatik/Medieninformatik/index.php.en)

E-Mail: github@stefantaubert.com

This project is licensed under the terms of the MIT license.

Please cite the paper in your publications if it helps your research.
```
@inproceedings{taubert2019automated,
  title={Automated Lifelog Moment Retrieval based on Image Segmentation and Similarity Scores},
  author={Taubert, Stefan and Kahl, Stefan and Kowerko, Danny and Eibl, Maximilian},
  booktitle={CLEF2019 Working Notes. CEUR Workshop Proceedings},
  pages={09--12},
  year={2019}
}
```
<b>You can download our working notes here:</b> [TUC MI Lifelog 2019 Working Notes PDF](http://ceur-ws.org/Vol-2380/paper_83.pdf)

## Get it running
![Python](https://img.shields.io/badge/python-3.6.9-green.svg)

### Step 1 - Download dataset

#### Prepare Folders

```bash
mkdir '/datasets/lifelog-2019'
cd '/datasets/lifelog-2019'
mkdir 'data'
mkdir 'dev/lmrt'
mkdir 'test/lmrt'
```

#### Download [dataset](https://www.crowdai.org/clef_tasks/9/task_dataset_files?challenge_id=62) into the data dir and run following commands:

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

The directory structure of `/datasets/lifelog-2019/data` should be the following:

```
minute_based_table
¦   u1.csv
¦   u2.csv
u1
¦   Autographer
¦   ¦   2018_05_03
¦   ¦   ¦   *.JPG
¦   ¦   ¦   ...
¦   ¦   ...
¦   u1_photos
¦   ¦   *.jpg
¦   ¦   ...
u2
¦   Autographer
¦   ¦   2018_05_09
¦   ¦   ¦   *.JPG
¦   ¦   ¦   ...
¦   ¦   ...
¦   u2_photos
¦   ¦   *.jpg
¦   ¦   ...
visual_concepts
¦   u1_categories_attr_concepts.csv
¦   u2_categories_attr_concepts.csv
```

#### Prepare queries and ground truth

Download files:

- `clusters.csv` to `/datasets/lifelog-2019/dev/lmrt/clusters.csv`
- `LMRT_gt.csv` to `/datasets/lifelog-2019/dev/lmrt/LMRT_gt.csv`
- `dev_topics.pdf` to `/datasets/lifelog-2019/dev/lmrt/topics.xml`
- `test_topics.pdf` to `/datasets/lifelog-2019/test/lmrt/topics.xml`

#### Create further folders

1. Create `/datasets/lifelog-2019/cache`
2. Create `/datasets/lifelog-2019/tmp`

### Step 2 - Set development environment

```bash
mkdir '/code/lifelog-2019'
cd '/code/lifelog-2019'
git clone git@github.com:stefantaubert/imageclef-lifelog-2019.git
pip install –r requirements.txt
nano src/io/data_dir_config.py
```

Now write the datapath from Step 1 into the file like so:

```py
root = "/datasets/lifelog-2019/"
```

On VS Code set env:

```json
"env": {"PYTHONPATH":"/code/lifelog-2019"}
```

You need Python 3.6.9. (separate environment is recommended).

### Step 3 - Getting Yolo

First install Cuda 10.1 if you want to run with GPU

```bash
mkdir '/datasets/lifelog-2019/res'
git clone https://github.com/pjreddie/darknet
cd darknet
wget https://pjreddie.com/media/files/yolov3.weights
nano Makefile
```

set GPU=1 and save

```bash
sudo make
```

Adjust the file `data/coco.data` so that names are the absolute location of `coco.names`.

### Step 4 - Getting Detectron

```bash
cd '/datasets/lifelog-2019/res'
git clone https://github.com/facebookresearch/Detectron.git
```

### Step 4 - Adjusting Errors in Dataset

File: `/datasets/lifelog-2019/data/minute_based_table/u2.csv`

Errors:

```
line 12080: expected 50 fields, saw 51 (u2_20180517_0918)
line 12081: expected 50 fields, saw 51 (u2_20180517_0919)
line 12145: expected 50 fields, saw 51 (u2_20180517_1023)
```

-> Remove one `"",` in the middle

Typos in directory `/datasets/lifelog-2019/data/u1/Autographer/2018_05_14/`:

- Adjust extention `B00002730_21I6X0_20180514_145029E.jpg` to `.JPG`

### Step 5 - Install XGBoost

For faster training install xgboost with gpu-support:
- on windows you find an intruction [here](http://www.picnet.com.au/blogs/guido/post/2016/09/22/xgboost-windows-x64-binaries-for-download/).
- on linux you find an instruction [here](https://github.com/dmlc/xgboost/blob/master/doc/build.md), td;tr:
    - if you have old version of cmake: deinstall old version and install new version from [here](https://cmake.org/download/) to e.g. `usr/local/` and create symlink `/usr/local/bin/cmake`
    - open terminal in e.g. `/datasets/lifelog-2019/res`

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

### Step 6 - Download NLTK data

Run `./tools/detecting/nltk_downloader.py`.

### Step 7 - Prepare GloVe Word Vectors

1. Download [glove.840B.300d](http://nlp.stanford.edu/data/glove.840B.300d.zip) vectors.
2. Unzip into `/datasets/lifelog-2019/res/glove`
3. Convert the data into a word2vec-Model with `./tools/detecting/GloVe_converter.py`.

### Step 8 - Predict further Labels

1. Execute `./tools/detecting/yolo/YoloDetectorImageNet.py`
1. Execute `./tools/detecting/yolo/YoloDetectorOpenImages.py`
1. Execute `./tools/detecting/detectron/DetectronDetector.py`

## Execute runs

You find all submitted runs in `./submissions/runs` which are all executable.
