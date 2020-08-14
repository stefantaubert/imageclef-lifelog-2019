from src.io.ReadingContext import ReadingContext
from src.data.DataBase import DataBase
from src.globals import vc_att_cols

name_sun = "SUNattributes"

class SUNattributesData(DataBase):
    def __init__(self, ctx: ReadingContext):
        return super().__init__(ctx)

    def get_name(self):
        return name_sun

    def __get_data_dict__(self):
        return self.ctx.vc_dict()
        
    def __get_label_columns__(self):
        return vc_att_cols
