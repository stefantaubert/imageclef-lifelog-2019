from src.detecting.DetectorBase import DetectorBase
from src.io.paths import get_path_darknet
from src.io.paths import get_path_yolo
from src.globals import usr1
from src.detecting.darknet import load_net
from src.detecting.darknet import load_meta
from src.detecting.darknet import detect

"""
The default threshold is set to 0.25
"""

class YoloDetector(DetectorBase):
    
    def __init__(self):
        darknet = get_path_darknet()
        self.encoding = 'utf-8'
        net = "{darknet}cfg/yolov3.cfg".format(darknet=darknet).encode(self.encoding)
        meta = "{darknet}cfg/coco.data".format(darknet=darknet).encode(self.encoding)
        weights = "{darknet}yolov3.weights".format(darknet=darknet).encode(self.encoding)
        self.net = load_net(net, weights, 0)
        self.meta = load_meta(meta)

    def detect_image(self, img):
        r = detect(self.net, self.meta, img.encode(self.encoding))
        return r

if __name__ == "__main__":
    d = YoloDetector()
    csv = d.detect_images_auto(usr1)
    file_name = get_path_yolo(usr=usr1)
    csv.to_csv(file_name, index=False)