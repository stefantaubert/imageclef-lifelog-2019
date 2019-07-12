from src.detecting.DetectorBase import DetectorBase
from src.io.paths import get_path_darknet
from src.io.paths import get_path_yolo_imgnet
from src.globals import usr1
from src.detecting.darknet import load_net
from src.detecting.darknet import load_meta
from src.detecting.darknet import classify
from src.detecting.darknet import detect
from src.detecting.darknet import load_image
from src.detecting.darknet import free_image

"""
Darknet53 448x448
"""

class YoloDetectorImageNet(DetectorBase):
    threshold = 0.5

    def __init__(self):
        darknet = get_path_darknet()
        self.encoding = 'utf-8'
        net = "{darknet}cfg/darknet53_448.cfg".format(darknet=darknet).encode(self.encoding)
        meta = "{darknet}cfg/imagenet1k.data".format(darknet=darknet).encode(self.encoding)
        weights = "{darknet}darknet53_448.weights".format(darknet=darknet).encode(self.encoding)
        self.net = load_net(net, weights, 0)
        self.meta = load_meta(meta)

    def detect_image(self, img):
        #r = detect(self.net, self.meta, img.encode(self.encoding), 0, 0, 0)
        im = load_image(img.encode(self.encoding), 0, 0)
        r = classify(self.net, self.meta, im)
        s = sum([ri[1] for ri in r])
        free_image(im)
        r = [(ri[0], ri[1], (0,0,0,0)) for ri in r if ri[1] >= self.threshold]
        return r

if __name__ == "__main__":
    usr = usr1
    d = YoloDetectorImageNet()
    csv = d.detect_images_auto(usr)
    file_name = get_path_yolo_imgnet(usr=usr)
    csv.to_csv(file_name, index=False)