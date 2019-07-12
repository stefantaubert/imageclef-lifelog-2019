import unittest

from src.vectoring.VectorBuilderBase import VectorBuilderBase

class DummyImageVectorBuilder(VectorBuilderBase):
    def build_vector(self, segment):
        return segment + ["test"]

class DummyQueryVectorBuilder(VectorBuilderBase):
    def build_vector(self, topic):
        return [topic["description"], "test"]

class UnitTests(unittest.TestCase):
    
    def test_build_image_vector(self):
        vb = DummyImageVectorBuilder()

        seg = ["Img1", "Img2", "Img3", "Img4", "Img5", "Img6"]

        vec = vb.build_vector(seg)

        self.assertEqual(seg + ["test"], vec)

    def test_build_image_vectors(self):
        vb = DummyImageVectorBuilder()

        segs = [["Img1"], ["Img2", "Img4"], ["Img3"]]

        vecs = vb.build_vectors(segs)

        assert_vecs = [
            ["Img1", "test"],
            ["Img2", "Img4", "test"],
            ["Img3", "test"],
        ]

        self.assertEqual(assert_vecs, vecs)

    def test_build_query_vector(self):
        vb = DummyQueryVectorBuilder()

        topic = {
            "description": "dummy"
        }

        vec = vb.build_vector(topic)

        self.assertEqual(["dummy", "test"], vec)

    def test_build_query_vectors(self):
        vb = DummyQueryVectorBuilder()

        tops = [
            {"description": "dummy"},
            {"description": "dummy2"},
        ]

        vec = vb.build_vectors(tops)

        self.assertEqual([["dummy", "test"], ["dummy2", "test"]], vec)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
