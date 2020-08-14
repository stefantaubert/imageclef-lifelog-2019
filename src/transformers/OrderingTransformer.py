from pdtransform import NoFitMixin
from collections import OrderedDict

def sort_dict(d: dict, reverse):
    s_items = sorted(d.items(), key=lambda t: t[1], reverse=reverse)
    res = OrderedDict(s_items)
    return res

class OrderingTransformer(NoFitMixin):
    def __init__(self, reverse=False):
            self.reversed = reverse

    def transform(self, submission: dict):
        return sort_dict(submission, self.reversed)