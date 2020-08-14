import pandas as pd

from tqdm import tqdm
from src.io.img_path_parser import get_paths

class DetectorBase():

    def detect_image(self, img):
        raise NotImplementedError()

    def detect_images(self, imgs):
        result = []

        for img in tqdm(imgs):
            pred = self.detect_image(img[1])
            row = [img[0]]
            for p in pred:
                bbox = p[2]
                bbox_str = "{x} {y} {w} {h}".format(x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3])
                concept = p[0]
                try:
                    concept = concept.decode('utf-8')
                except AttributeError:
                    pass
                row.extend([concept, p[1], bbox_str])
            result.append(row)

        return result

    def detect_images_df(self, imgs):
        print("Predicting...")
        rows = self.detect_images(imgs)
        cols = ["image_id"]

        if len(rows) > 0:
            col_count = max(len(r) for r in rows)
            col_count = int((col_count - 1) / 3)
            
            for i in range(col_count):
                current_nr = str(i + 1).zfill(2)
                cols.append("concept_class_top{nr}".format(nr=current_nr))
                cols.append("concept_score_top{nr}".format(nr=current_nr))
                cols.append("concept_bbox_top{nr}".format(nr=current_nr))
        
        res = pd.DataFrame(columns=cols, data=rows)
        print("Finished.")
        return res

    def detect_images_auto(self, usr=1):
        paths = get_paths(usr=usr)
        res = self.detect_images_df(paths)
        return res