from pdtransform import NoFitMixin
from collections import OrderedDict

def get_day_from_img_id(img_id):
    return img_id[3:11]

def rearrange(subm: OrderedDict, per_day):
    assert isinstance(subm, OrderedDict)
    if per_day == 0:
        return subm
    
    ks = list(subm.keys())
    days = [get_day_from_img_id(k) for k in ks]
    unique_days = list(OrderedDict.fromkeys(days))
    res = OrderedDict({d: [] for d in unique_days})
    for i in range(len(ks)):
        k = ks[i]
        day = days[i]
        res[day].append(k)
    
    result = OrderedDict()
    while len(result) != len(ks):
        for k in res.keys():
            imgs: list = res[k]
            for _ in range(per_day):
                if len(imgs) > 0:
                    img_id = imgs[0]
                    result[img_id] = subm[img_id]
                    del res[k][0]
    return result

class RearrangeTransformer(NoFitMixin):
    def __init__(self, n):
        self.n = n
    
    def transform(self, submission: OrderedDict):
        return rearrange(submission, self.n)