from tqdm import tqdm

class VectorBuilderBase():
    
    def build_vector(self, input):
        raise NotImplementedError()

    def build_vectors(self, input_arr):
        res = []
        for i in tqdm(input_arr):
            vec = self.build_vector(i)
            res.append(vec)
        return res