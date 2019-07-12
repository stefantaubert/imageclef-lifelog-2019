from src.io.ReadingContext import ReadingContext
from src.data.DataBase import DataBase

name_oi = "openimages"

class OpenImagesData(DataBase):
    def __init__(self, ctx: ReadingContext):
        return super().__init__(ctx)

    def get_name(self):
        return name_oi

    def __get_data_dict__(self):
        return self.ctx.openimages().set_index("image_id").to_dict('index')
        
    def __get_label_columns__(self):
        return [col for col in self.ctx.openimages().columns if "class" in col]

    def __get_score_columns__(self):
        return [col for col in self.ctx.openimages().columns if "score" in col]
