import numpy as np

from src.data.DataBase import DataBase
from src.vectoring.VectorBuilderBase import VectorBuilderBase

class ImageVectorBuilder(VectorBuilderBase):
    def __init__(self, extracted_data: DataBase):
        extracted_data.assert_is_extracted()
        self.__data__ = extracted_data
        self.__labels_c__ = len(self.__data__.labels)
    
    def build_img_vector(self, img_id):
        vec = np.zeros(self.__labels_c__)
        index = self.__data__.rows_img_id.index(img_id)
        row_labels = self.__data__.rows_labels[index]
        row_scores = self.__data__.rows_scores[index]

        for label_i, label in enumerate(row_labels):
            score = row_scores[label_i]
            label_vec_ind = self.__data__.labels.index(label)
            old_score = vec[label_vec_ind]
            vec[label_vec_ind] = max(old_score, score)

        return np.array(vec)

    def build_vector(self, segment):
        resulting_vector = np.zeros(self.__labels_c__)
        for img_id in segment:
            vec = self.build_img_vector(img_id)
            resulting_vector = np.array([max(vec[i], resulting_vector[i]) for i in range(self.__labels_c__)])
        return resulting_vector
