import unittest

from src.globals import usr1
from src.globals import usr2
from src.globals import ds_dev
from src.globals import ds_test
from src.globals import vc_cols_c
from src.globals import mbt_cols_c
from src.globals import top_id
from src.globals import top_idi
from src.globals import top_type
from src.globals import top_usr
from src.globals import top_title
from src.globals import top_desc
from src.globals import top_narrative
from src.io.reader import read_csv
from src.io.reader import read_vc
from src.io.reader import read_mbt
from src.io.reader import read_tops
from src.io.reader import read_tops_prep
from src.io.reader import read_clusters
from src.io.reader import read_gt
from src.io.reader import read_xml
from src.io.reader import read_att
from src.io.reader import read_con
from src.io.reader import read_cat
from src.io.reader import read_vec
from src.io.reader import read_yolo
from src.io.reader import read_yolo_imgnet
from src.io.reader import read_detectron

class UnitTests(unittest.TestCase):
    
    def test_read_csv(self):
        path = "./tests/testdata/test.csv"
        mbt = read_csv(path)
        self.assertEqual(3, len(mbt.columns))

    def test_read_xml(self):
        path = "./tests/testdata/tops.xml"
        xml = read_xml(path)
        self.assertEqual(2, len(xml))
        first = xml[0]
        self.assertEqual("0045", first[top_id])
        self.assertEqual(45, first[top_idi])
        self.assertEqual("Test2", first[top_type])
        self.assertEqual("Test3 Test31", first[top_usr])
        self.assertEqual("Test4 Test41", first[top_title])
        self.assertEqual("Test5", first[top_desc])
        self.assertEqual("Test6", first[top_narrative])

    def test_read_vec(self):
        model = read_vec(big=False)
        self.assertGreater(len(model.vocab), 0)

    def test_read_att(self):
        att = read_att()
        self.assertEqual(type(list()), type(att))
        self.assertEqual(sorted(set(att)), att)
        self.assertEqual(102, len(att))

    def test_read_con(self):
        con = read_con()
        self.assertEqual(type(set()), type(con))
        self.assertEqual(set(sorted(con)), con)
        self.assertEqual(183, len(con))

    def test_read_cat(self):
        cat = read_cat()
        self.assertEqual(type(set()), type(cat))
        self.assertEqual(set(sorted(cat)), cat)
        self.assertEqual(365, len(cat))

    def test_read_tops_prep(self):
        tops = read_tops_prep()
        self.assertEqual(4, len(tops.columns))

    def test_read_detectron_u1(self):
        detectron = read_detectron(usr=usr1)
        self.assertGreater(0, len(detectron.columns))

    def test_read_yolo_imgnet_u1(self):
        yolo = read_yolo_imgnet(usr=usr1)
        self.assertGreater(0, len(yolo.columns))

    def test_read_yolo_u1(self):
        yolo = read_yolo(usr=usr1)
        self.assertGreater(0, len(yolo.columns))

    def test_read_vc_default(self):
        vc = read_vc()
        self.assertEqual(vc_cols_c, len(vc.columns))

    def test_read_vc_u1_versNone(self):
        vc = read_vc(usr=usr1, vers=None)
        self.assertEqual(vc_cols_c, len(vc.columns))

    def test_read_vc_u2_versNone(self):
        vc = read_vc(usr=usr2, vers=None)
        self.assertEqual(vc_cols_c, len(vc.columns))

    def test_read_mbt_default(self):
        mbt = read_mbt()
        self.assertEqual(mbt_cols_c, len(mbt.columns))
    
    def test_read_mbt_u1_versNone(self):
        mbt = read_mbt(usr=usr1, vers=None)
        self.assertEqual(mbt_cols_c, len(mbt.columns))
    
    def test_read_mbt_u2_versNone(self):
        mbt = read_mbt(usr=usr2, vers=None)
        self.assertEqual(mbt_cols_c, len(mbt.columns))
    
    def test_read_clusters(self):
        clusters = read_clusters()
        self.assertEqual(3, len(clusters.columns))
    
    def test_read_gt(self):
        gt = read_gt()
        self.assertEqual(3, len(gt.columns))
    
    def test_read_tops_default(self):
        tops = read_tops()
        self.assertEqual(10, len(tops))

    def test_read_tops_dev(self):
        tops = read_tops(1)
        self.assertEqual(10, len(tops))

    def test_read_tops_test(self):
        #not available jet
        pass
        #tops = read_tops(2)
        #self.assertEqual(10, len(tops))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)