from pdtransform import NoFitMixin
from pathlib import Path
import pickle
from src.io.paths import get_dir_cache
from src.common.helper import create_parent_dir

class CachableTransformerBase(NoFitMixin):
    def __init__(self, usr: int, is_dirty=True, suffix=""):
        self.is_dirty = is_dirty
        self.suffix = suffix
        self.usr = usr
        self.path = "{cachedir}{name}{suffix}.pkl".format(cachedir=get_dir_cache(usr=self.usr), name=self.__class__.__name__, suffix=self.suffix)

    def transform_core(self, X):
        raise NotImplementedError()

    def from_cache(self):
        assert Path(self.path).exists()
        return pickle.load(open(self.path, "rb"))

    def transform(self, X):
        self.before_transform(X)
        if not self.is_dirty and Path(self.path).exists():
            self.result = self.from_cache()
        else:
            self.result = self.transform_core(X)
            create_parent_dir(self.path)
            pickle.dump(self.result, open(self.path, "wb"))
        self.after_transform(self.result)
        return self.result
    
    def before_transform(self, X):
        pass
    
    def after_transform(self, X):
        pass

if __name__ == "__main__":
    from sklearn.pipeline import Pipeline

    class DummyTrans(CachableTransformerBase):
        def __init__(self, is_dirty=True):
            return super().__init__(is_dirty=is_dirty)

        def transform_core(self, X):
            return [X, X, X]

    pipe = Pipeline([
        ("dummy", DummyTrans()),
    ])
    
    res = pipe.transform("ulf")
    print(res)
    