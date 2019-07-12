from src.io.ReadingContext import ReadingContext
from src.common.idf_tools import get_idf
from src.query_translation.get_label_tokens import get_label_tokens
from src.data.read_unified import read_unified

# NOTE: np.flatten works only for same shaped list of lists
def __flatten__(lst: list):
    return [y for x in lst for y in x]

class DataBase():
    def __init__(self, ctx: ReadingContext):
        self.ctx = ctx
        self.__extracted__ = False

    def get_name(self):
        raise NotImplementedError()

    def __unify__(self, word):
        return word

    def __get_score_columns__(self):
        return []

    def __get_data_dict__(self):
        raise NotImplementedError()

    def __get_label_columns__(self):
        raise NotImplementedError()

    def __get_data__(self): 
        try:
            return self.__data__
        except:
            self.__data__ = {
                "columns": self.__get_label_columns__(),
                "score_columns": self.__get_score_columns__(),
                "predictions": self.__get_data_dict__(),
                "unification_method": self.__unify__,
            }

            return self.__data__

    def assert_is_extracted(self):
        assert self.__extracted__

    def extract_data(self, threshold: float, optimize_labels: bool):
        self.rows_img_id, self.rows_labels, self.rows_scores = read_unified(self.__get_data__(), threshold, self.ctx.vocab(), optimize_labels)
        self.labels = sorted(set(__flatten__(self.rows_labels)))
        self.tokenized_labels = get_label_tokens(self.labels, self.ctx.vocab())
        self.idfs_dict = get_idf(self.rows_labels)
        self.idfs = [self.idfs_dict[label] for label in self.labels]
        self.__extracted__ = True
        print("{name}: extracted {n} labels @ threshold {t}.".format(name=self.get_name(), n=len(self.labels), t=str(threshold)))