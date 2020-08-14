from pdtransform import NoFitMixin

def flatten(array):
    res = []
    for itm in array:
        res.extend(itm)
    return res

class FlatteningTransformer(NoFitMixin):
    """
    """
    def transform(self, days_segments: list):
        print("Flattening...")
        flattend = flatten(days_segments)
        print("Got {c} elements after flattening.".format(c=str(len(flattend))))
        return flattend