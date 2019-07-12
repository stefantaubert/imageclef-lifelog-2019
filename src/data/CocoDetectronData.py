from src.io.ReadingContext import ReadingContext
from src.data.DataBase import DataBase
from src.data.unify_coco import unify_coco

name_detectron = "detectron"

class CocoDetectronData(DataBase):
    def __init__(self, ctx: ReadingContext):
        return super().__init__(ctx)

    def get_name(self):
        return name_detectron

    def __unify__(self, word):
        word = unify_coco(word)
        return word

    def __get_data_dict__(self):
        return self.ctx.detectron().set_index("image_id").to_dict('index')
        
    def __get_label_columns__(self):
        return [col for col in self.ctx.detectron().columns if "class" in col]

    def __get_score_columns__(self):
        return [col for col in self.ctx.detectron().columns if "score" in col]
        