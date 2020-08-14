from src.io.ReadingContext import ReadingContext
from src.data.DataBase import DataBase
from src.globals import vc_cat_cols
from src.globals import vc_cat_score_cols

name_places = "places365"

class Places365Data(DataBase):
    def __init__(self, ctx: ReadingContext):
        return super().__init__(ctx)

    def get_name(self):
        return name_places

    def __unify__(self, word):
        cat = word.replace('_', ' ').replace('/', ' ')
        #cat = cat.replace("outdoor", "").replace("indoor", "").strip()
        return cat

    def __get_data_dict__(self):
        return self.ctx.vc_dict()
        
    def __get_label_columns__(self):
        return vc_cat_cols

    def __get_score_columns__(self):
        return vc_cat_score_cols
