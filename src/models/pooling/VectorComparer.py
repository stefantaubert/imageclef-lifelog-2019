from src.data.DataBase import DataBase
from src.vectoring.ImageVectorBuilder import ImageVectorBuilder
from src.vectoring.SimilarityVectorBuilder import SimilarityVectorBuilder
from src.comparing.get_similarities import get_similarities
from src.models.pooling.data_opts_to_comparer_opts_converter import convert_to_comparing_opts

class VectorComparer():
    def __init__(self, opts: dict, emb, imgs: list, query_tokens: list, data: DataBase):
        data.assert_is_extracted()
        self.data = data
        self.opts = opts
        self.imgs = imgs
        self.emb = emb
        self.query_tokens = query_tokens

    def __extract_que_vecs__(self):
        que_vb = SimilarityVectorBuilder(self.data.tokenized_labels, self.emb)    
        self.que_vecs = que_vb.build_vectors(self.query_tokens)

    def __extract_img_vecs__(self):
        img_vb = ImageVectorBuilder(self.data)
        self.img_vecs = img_vb.build_vectors(self.imgs)

    def fit(self):
        print("Extract query vectors...")
        self.__extract_que_vecs__()

        print("Extract image vectors...")
        self.__extract_img_vecs__()
    
    def compare(self):
        # must be converted here because the data_opts can change after fitting in the experiments
        cmp_opts = convert_to_comparing_opts(self.opts, self.data.get_name())
        self.similarities = get_similarities(self.img_vecs, self.que_vecs, self.data.idfs, cmp_opts)
