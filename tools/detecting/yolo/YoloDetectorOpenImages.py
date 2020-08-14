from src.detecting.DetectorBase import DetectorBase
from src.io.paths import get_path_darknet
from src.io.paths import get_path_yolo_openimgs
from src.globals import usr1
from src.detecting.darknet import load_net
from src.detecting.darknet import load_meta
from src.detecting.darknet import detect

"""
"""

class YoloDetectorOpenImages(DetectorBase):
    
    def __init__(self):
        darknet = get_path_darknet()
        self.encoding = 'utf-8'
        net = "{darknet}cfg/yolov3-openimages.cfg".format(darknet=darknet).encode(self.encoding)
        meta = "{darknet}cfg/openimages.data".format(darknet=darknet).encode(self.encoding)
        weights = "{darknet}yolov3-openimages.weights".format(darknet=darknet).encode(self.encoding)
        self.net = load_net(net, weights, 0)
        self.meta = load_meta(meta)

    def detect_image(self, img):
        r = detect(self.net, self.meta, img.encode(self.encoding))
        #print(r)
        return r

if __name__ == "__main__":
    d = YoloDetectorOpenImages()
    csv = d.detect_images_auto(usr1)
    file_name = get_path_yolo_openimgs(usr=usr1)
    csv.to_csv(file_name, index=False)