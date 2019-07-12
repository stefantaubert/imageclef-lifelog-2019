import pandas as pd

from src.io.ReadingContext import ReadingContext
from src.globals import mbt_id_cols
from src.data.DataBase import DataBase

name_activity = "activity"

class ActivityData(DataBase):
    def __init__(self, ctx: ReadingContext):
        return super().__init__(ctx)

    def get_name(self):
        return name_activity

    def __get_label_columns__(self):
        return ["activity"]

    def __get_data_dict__(self):
        vals = []
        ids = []
        for _, row in self.ctx.mbt_dict().items():
            val = row["activity"]
            for col in mbt_id_cols:
                img_id = row[col]
                if img_id and img_id in self.ctx.vc_dict().keys():
                    vals.append(val)
                    ids.append(img_id)
        df = pd.DataFrame({"image_id": ids, "activity": vals})
        return df.set_index("image_id").to_dict('index')
