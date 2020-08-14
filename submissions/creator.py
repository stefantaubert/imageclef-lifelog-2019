from src.io.paths import get_dir_subm
from src.common.helper import create_dir_if_not_exists
from submissions.converter import subm_to_df
import datetime

def create_submission(subm, name):
    dest_dir = get_dir_subm()
    create_dir_if_not_exists(dest_dir)
    timestr = datetime.datetime.now().strftime("%d%m%y_%H%M%S")
    dest_name = "{d}{n}_{t}.csv".format(d=dest_dir, n=name, t=timestr)
    df = subm_to_df(subm)
    df.to_csv(dest_name, index=False, header=False)
    return dest_name