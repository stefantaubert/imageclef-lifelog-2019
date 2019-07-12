from pdtransform import NoFitMixin

class PotentialTransformer(NoFitMixin):
    def __init__(self, n, m):
        self.n = n
        self.m = m

    def transform(self, vector):
        res = vector * self.n
        res = res ** self.m
        return res