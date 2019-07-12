import pandas as pd

from src.io.ReadingContext import ReadingContext
from src.data.DataBase import DataBase
from src.globals import mbt_id_cols

name_loc = "location"

__common_labels__ = [
    "solicitor",
    "hotel",
    "bar",
    "restaurant",
    "railway station",
    "home",
    "store",
    "shopping centre",
    "university",
    "dentist",
    "work",
    "airport",
    "bakery",
    "cafe",
    "college",
    "embassy",
    "costa coffee",
]

def __get_mapped_location__(location):
    for common_label in __common_labels__:
        if common_label in location.lower():
            return common_label
    return ""

class LocationData(DataBase):
    def __init__(self, ctx: ReadingContext):
        return super().__init__(ctx)

    def get_name(self):
        return name_loc

    def __unify__(self, word):
        return __get_mapped_location__(word)

    def __get_data_dict__(self):
        times = []
        ids = []
        for _, row in self.ctx.mbt_dict().items():
            time = row["name"]
            for col in mbt_id_cols:
                img_id = row[col]
                if img_id and img_id in self.ctx.vc_dict().keys():
                    times.append(time)
                    ids.append(img_id)
        df = pd.DataFrame({"image_id": ids, "location": times})
        return df.set_index("image_id").to_dict('index')
        
    def __get_label_columns__(self):
        return ["location"]
