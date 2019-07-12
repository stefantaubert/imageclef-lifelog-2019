from pdtransform import NoFitMixin

def wrap(array):
    res = [[itm] for itm in array]
    return res

class WrappingTransformer(NoFitMixin):
    """
    """
    def transform(self, array: list):
        return wrap(array)