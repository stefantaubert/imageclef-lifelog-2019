from src.segmentation.CachableTransformerBase import CachableTransformerBase

def extract_days(auto_img_paths: list):
    chronologic_sorted = sorted(auto_img_paths)
    # .../2018_05_03/B00001387_21I6X0_20180503_072550E.JPG
    res = {}
    for path in chronologic_sorted:
        day_str = path.split('/')[-2] #2018_05_03
        if day_str in res.keys():
            res[day_str].append(path)
        else:
            res[day_str] = [path]

    days = [res[k] for k in sorted(res.keys())]
    return days

class DaysExtractionTransformer(CachableTransformerBase):
    """
    Maps the images to the day in which they were shooted.
    Input: Paths to all autographer images of a user
    Output: list of Images per day as a list
    """
    def __init__(self, usr: int, is_dirty: bool = False):
        return super().__init__(usr=usr, is_dirty=is_dirty)
        
    def before_transform(self, _):
        print("Extracting days...")
        
    def transform_core(self, auto_img_paths: list):
        days = extract_days(auto_img_paths)
        return days

if __name__ == "__main__":
    from sklearn.pipeline import Pipeline
    from src.segmentation.VCReaderTransformer import VCReaderTransformer
    from src.segmentation.AutographerImageExtractorTransformer import AutographerImageExtractorTransformer

    usr = 1
    pipe = Pipeline([
        ("read_vc", VCReaderTransformer()),
        ("extract_auto_imgs", AutographerImageExtractorTransformer(usr)),
        ("extract_days", DaysExtractionTransformer()),
    ])
    
    res = pipe.transform(usr)
    print(len(res))
    