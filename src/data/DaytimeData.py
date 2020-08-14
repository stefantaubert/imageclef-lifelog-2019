import pandas as pd

from src.io.ReadingContext import ReadingContext
from src.data.DataBase import DataBase
from src.globals import mbt_id_cols

name_daytime = "daytime"

__day_times__ = {
    "morning": [x for x in range(400, 1200)],
    "afternoon": [x for x in range(1200, 1700)],
    "evening": [x for x in range(1700, 2200)],
    "night": [x for x in range(2200, 2400)] + [x for x in range(0, 400)],
}

def __get_day_time__(img_id):
    midnight = ""
    time = img_id[9:].lstrip("0")
    time = 0 if time == midnight else int(time)

    for daytime, time_range in __day_times__.items():
        if time in time_range: 
            return daytime

    assert False

class DaytimeData(DataBase):
    def __init__(self, ctx: ReadingContext):
        return super().__init__(ctx)

    def get_name(self):
        return name_daytime

    def __unify__(self, img_id):
        return __get_day_time__(img_id)

    def __get_data_dict__(self):
        times = []
        ids = []
        for _, row in self.ctx.mbt_dict().items():
            time = row["local_time"]
            for col in mbt_id_cols:
                img_id = row[col]
                if img_id and img_id in self.ctx.vc_dict().keys():
                    times.append(time)
                    ids.append(img_id)
        df = pd.DataFrame({"image_id": ids, "local_time": times})
        return df.set_index("image_id").to_dict('index')
        
    def __get_label_columns__(self):
        return ["local_time"]
