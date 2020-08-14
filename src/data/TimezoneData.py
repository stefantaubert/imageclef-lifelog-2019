import pandas as pd

from src.io.ReadingContext import ReadingContext
from src.data.DataBase import DataBase
from src.globals import mbt_id_cols

name_timezone = "timezone"

def __get_time_zone_city__(timezone):
    if timezone and timezone != "None":
        splitted = timezone.split("/")
        if len(splitted) < 2:
            print(splitted)
            assert False
        city = splitted[1]
        return city
    else:
        return ""

class TimezoneData(DataBase):
    def __init__(self, ctx: ReadingContext):
        return super().__init__(ctx)

    def get_name(self):
        return name_timezone

    def __unify__(self, timezone):
        return __get_time_zone_city__(timezone)

    def __get_data_dict__(self):
        times = []
        ids = []
        for _, row in self.ctx.mbt_dict().items():
            time = row["time_zone"]
            for col in mbt_id_cols:
                img_id = row[col]
                if img_id and img_id in self.ctx.vc_dict().keys():
                    times.append(time)
                    ids.append(img_id)
        df = pd.DataFrame({"image_id": ids, "timezone": times})
        return df.set_index("image_id").to_dict('index')
        
    def __get_label_columns__(self):
        return ["timezone"]
