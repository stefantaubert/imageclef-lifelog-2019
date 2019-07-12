from src.io.ReadingContext import ReadingContext
from src.data.DataBase import DataBase

name_imagenet = "imagenet"

class ImageNetData(DataBase):
    def __init__(self, ctx: ReadingContext):
        return super().__init__(ctx)

    def get_name(self):
        return name_imagenet

    def __get_data_dict__(self):
        return self.ctx.imagenet().set_index("image_id").to_dict('index')

    def __get_label_columns__(self):
        return [col for col in self.ctx.imagenet().columns if "class" in col]

    def __get_score_columns__(self):
        return [col for col in self.ctx.imagenet().columns if "score" in col]
