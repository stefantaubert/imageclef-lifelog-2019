import pickle

from src.detecting.DetectorBase import DetectorBase
from src.detecting.img_path_parser import get_paths
from src.io.paths import get_path_detectron
from src.globals import usr1

"""
I run detectron on remote machine and save the results as a dictionary with keys = image_id and values: class predictions.
This dictionary will be loaded as model.
The threshold is set to 0.5
"""

class DetectronDetector(DetectorBase):
    def __init__(self):
        print("Load predictions from detectron...")
        self.predictions = pickle.load(open( "/tmp/pred.pkl", "rb"), encoding='latin1')
        print("Preprocessing...")
        paths = get_paths()
        self.ids_to_paths = {}
        for path in paths:
            img_id = path[0]
            img_path = path[1]
            self.ids_to_paths[img_path] = img_id
        
    def detect_image(self, img):
        img_id = self.ids_to_paths[img]
        if img_id in self.predictions:
            r = self.predictions[img_id]
        else:
            r = []
        return r

if __name__ == "__main__":
    d = DetectronDetector()
    csv = d.detect_images_auto(usr1)
    file_name = get_path_detectron(usr=usr1)
    csv.to_csv(file_name, index=False)
    print("Successfully saved to", file_name)