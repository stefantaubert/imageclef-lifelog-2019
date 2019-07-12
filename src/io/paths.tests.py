import unittest

from src.io.data_dir_config import root
from src.io.paths import get_path_vc
from src.io.paths import get_path_mbt
from src.io.paths import get_path_tops
from src.io.paths import get_path_tops_prep
from src.io.paths import get_path_clusters
from src.io.paths import get_path_gt
from src.io.paths import get_path_labels
from src.io.paths import get_dir_auto
from src.io.paths import get_dir_pers
from src.io.paths import get_dir_img
from src.io.paths import get_dir_cache
from src.io.paths import get_dir_subm
from src.io.paths import get_path_att
from src.io.paths import get_path_cat
from src.io.paths import get_path_con
from src.io.paths import get_path_vec
from src.io.paths import get_path_darknet
from src.io.paths import get_path_from_rel_img_path
from src.io.paths import get_path_yolo
from src.io.paths import get_path_yolo_imgnet
from src.io.paths import get_path_detectron
from src.io.paths import get_path_sgmts

class UnitTests(unittest.TestCase):
    
    def test_get_path_from_rel_img_path_personal(self):
        root = "/unittest/"
        rel_path = "u2_photos/2018-05-10 17.32.02.jpg"
        path = get_path_from_rel_img_path(rel_path, usr=2, root=root)
        assertPath = "{root}data/u2/u2_photos/2018-05-10 17.32.02.jpg".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_from_rel_img_path_autographer(self):
        root = "/unittest/"
        rel_path = "2018_05_10/B00005018_21I6X0_20180510_173036E.JPG"
        path = get_path_from_rel_img_path(rel_path, usr=2, root=root)
        assertPath = "{root}data/u2/Autographer/2018_05_10/B00005018_21I6X0_20180510_173036E.JPG".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_darknet_default(self):
        path = get_path_darknet()
        assertPath = "{root}res/darknet/".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_labels_default(self):
        path = get_path_labels()
        assertPath = "{root}res/labels.pkl".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_vec_default(self):
        path = get_path_vec()
        assertPath = "{root}res/glove.6B/glove.6B.50d.w2v.txt".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_sgmts(self):
        path = get_path_sgmts(usr=2)
        assertPath = "{root}res/u2_segments.pkl".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_dir_subm(self):
        path = get_dir_subm()
        assertPath = "{root}submissions/".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_dir_cache(self):
        path = get_dir_cache(usr=2)
        assertPath = "{root}cache/u2/".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_cat_default(self):
        path = get_path_cat()
        assertPath = "{root}res/categories.txt".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_con_default(self):
        path = get_path_con()
        assertPath = "{root}res/concepts.txt".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_att_default(self):
        path = get_path_att()
        assertPath = "{root}res/attributes.mat".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_att(self):
        root = "/unittest/"
        path = get_path_att(root=root)
        assertPath = "{root}res/attributes.mat".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_detectron(self):
        root = "/unittest/"
        path = get_path_detectron(usr=2, root=root)
        assertPath = "{root}data/visual_concepts/u2_detectron.csv".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_yolo_imgnet(self):
        root = "/unittest/"
        path = get_path_yolo_imgnet(usr=2, root=root)
        assertPath = "{root}data/visual_concepts/u2_yolo_imagenet1k.csv".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_yolo(self):
        root = "/unittest/"
        path = get_path_yolo(usr=2, root=root)
        assertPath = "{root}data/visual_concepts/u2_yolo.csv".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_vc_default(self):
        path = get_path_vc()
        assertPath = "{root}data/visual_concepts/u1_categories_attr_concepts.csv".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_vc_u1_versNone(self):
        root = "/unittest/"
        path = get_path_vc(root=root, usr=1, vers=None)
        self.assertEqual("/unittest/data/visual_concepts/u1_categories_attr_concepts.csv", path)

    def test_get_path_vc_u1_vers0(self):
        root = "/unittest/"
        path = get_path_vc(root=root, usr=1, vers=0)
        self.assertEqual("/unittest/data/visual_concepts/u1_categories_attr_concepts_0.csv", path)

    def test_get_path_vc_u2_versNone(self):
        root = "/unittest/"
        path = get_path_vc(root=root, usr=2, vers=None)
        self.assertEqual("/unittest/data/visual_concepts/u2_categories_attr_concepts.csv", path)

    def test_get_path_vc_u2_vers0(self):
        root = "/unittest/"
        path = get_path_vc(root=root, usr=2, vers=0)
        self.assertEqual("/unittest/data/visual_concepts/u2_categories_attr_concepts_0.csv", path)

    def test_get_path_mbt_default(self):
        path = get_path_mbt()
        assertPath = "{root}data/minute_based_table/u1.csv".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_mbt_u1_versNone(self):
        root = "/unittest/"
        path = get_path_mbt(root=root, usr=1, vers=None)
        self.assertEqual("/unittest/data/minute_based_table/u1.csv", path)

    def test_get_path_mbt_u1_vers0(self):
        root = "/unittest/"
        path = get_path_mbt(root=root, usr=1, vers=0)
        self.assertEqual("/unittest/data/minute_based_table/u1_0.csv", path)

    def test_get_path_mbt_u2_versNone(self):
        root = "/unittest/"
        path = get_path_mbt(root=root, usr=2, vers=None)
        self.assertEqual("/unittest/data/minute_based_table/u2.csv", path)

    def test_get_path_mbt_u2_vers0(self):
        root = "/unittest/"
        path = get_path_mbt(root=root, usr=2, vers=0)
        self.assertEqual("/unittest/data/minute_based_table/u2_0.csv", path)

    def test_get_path_tops_prep(self):
        root = "/unittest/"
        path = get_path_tops_prep(root=root)
        self.assertEqual("/unittest/dev/lmrt/topics_prep.csv", path)

    def test_get_path_tops_default(self):
        path = get_path_tops()
        assertPath = "{root}dev/lmrt/topics.xml".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_tops_dev(self):
        root = "/unittest/"
        path = get_path_tops(root=root, ds=1)
        self.assertEqual("/unittest/dev/lmrt/topics.xml", path)

    def test_get_path_tops_test(self):
        root = "/unittest/"
        path = get_path_tops(root=root, ds=2)
        self.assertEqual("/unittest/test/lmrt/topics.xml", path)

    def test_get_path_clusters_default(self):
        path = get_path_clusters()
        assertPath = "{root}dev/lmrt/clusters.csv".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_clusters(self):
        root = "/unittest/"
        path = get_path_clusters(root=root)
        self.assertEqual("/unittest/dev/lmrt/clusters.csv", path)

    def test_get_path_gt_default(self):
        path = get_path_gt()
        assertPath = "{root}dev/lmrt/LMRT_gt.csv".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_path_gt(self):
        root = "/unittest/"
        path = get_path_gt(root=root)
        self.assertEqual("/unittest/dev/lmrt/LMRT_gt.csv", path)

    def test_get_dir_auto_default(self):
        path = get_dir_auto()
        assertPath = "{root}data/u1/Autographer/".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_dir_auto_u1(self):
        root = "/unittest/"
        path = get_dir_auto(root=root, usr=1)
        self.assertEqual("/unittest/data/u1/Autographer/", path)

    def test_get_dir_auto_u2(self):
        root = "/unittest/"
        path = get_dir_auto(root=root, usr=2)
        self.assertEqual("/unittest/data/u2/Autographer/", path)

    def test_get_dir_pers_default(self):
        path = get_dir_pers()
        assertPath = "{root}data/u1/u1_photos/".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_dir_pers_u1(self):
        root = "/unittest/"
        path = get_dir_pers(root=root, usr=1)
        self.assertEqual("/unittest/data/u1/u1_photos/", path)

    def test_get_dir_pers_u2(self):
        root = "/unittest/"
        path = get_dir_pers(root=root, usr=2)
        self.assertEqual("/unittest/data/u2/u2_photos/", path)

    def test_get_dir_img_default(self):
        path = get_dir_img()
        assertPath = "{root}data/u1/".format(root=root)
        self.assertEqual(assertPath, path)

    def test_get_dir_img_u1(self):
        root = "/unittest/"
        path = get_dir_img(root=root, usr=1)
        self.assertEqual("/unittest/data/u1/", path)

    def test_get_dir_img_u2(self):
        root = "/unittest/"
        path = get_dir_img(root=root, usr=2)
        self.assertEqual("/unittest/data/u2/", path)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)