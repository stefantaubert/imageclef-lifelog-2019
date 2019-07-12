from src.segmentation.CachableTransformerBase import CachableTransformerBase

def select_first(segments):
    res = []
    for segment in segments:
        assert len(segment) > 0
        res.append([segment[0]])
    return res

def select_last(segments):
    res = []
    for segment in segments:
        assert len(segment) > 0
        res.append([segment[-1]])
    return res

def select_first_last(segments):
    res = []
    for segment in segments:
        assert len(segment) > 1
        res.append([segment[0], segment[-1]])
    return res

def select_all(segments):
    return segments

class RepresentativeImageTransformer(CachableTransformerBase):
    """
    Parameters:
    - method: the method for the selection of the representative image
    Input: unordered segments (list of list of images)
    Output: representative pictures of each segment are token
    """
    def __init__(self, usr: int, method="first", is_dirty: bool = False):
        self.method = method
        return super().__init__(usr=usr, is_dirty=is_dirty)
        
    def before_transform(self, segments):
        img_count = sum([len(segment) for segment in segments])
        self.old_c = img_count
        print("Selecting representative images with method {m} out of {c} images...".format(m=self.method, c=str(img_count)))

    def transform_core(self, segments: list):
        if self.method == "first":
            return select_first(segments)
        elif self.method == "last":
            return select_last(segments)
        elif self.method == "first_last":
            return select_first_last(segments)
        elif self.method == "all":
            return select_all(segments)
        else:
            raise NotImplementedError()

    def after_transform(self, representatives):
        img_c = sum([len(r) for r in representatives])
        reduced_factor = round(100-(img_c/self.old_c*100),0)
        print("Selected {c} representatives, reduced image amount by {p}%.".format(c=str(img_c), p=str(reduced_factor)))