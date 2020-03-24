'''Test extraction of SIFT keypoints'''
import numpy as np
import unittest
import cudasift


class TestExtractKeypoints(unittest.TestCase):

    def test_01_01_nothing(self):
        # "How? Nothing will come of nothing.", Lear 1:1
        #
        data = cudasift.PySiftData(100)
        img = np.zeros((100, 100), np.uint8)
        cudasift.ExtractKeypoints(img, data)
        self.assertEqual(len(data), 0)
        df, keypoints = data.to_data_frame()
        self.assertEqual(len(df), 0)
        self.assertSequenceEqual(keypoints.shape, (0, 128))
        for column in "xpos", "ypos", "scale", "sharpness", "edgeness",\
                "orientation", "score", "ambiguity", "match", "match_xpos", \
                "match_ypos", "match_error", "subsampling":
            assert column in df.columns

    def test_01_02_speak_again(self):
        #
        # Check the four corners of a square
        #
        data = cudasift.PySiftData(100)
        img = np.zeros((100, 100), np.uint8)
        img[10:-9, 10] = 128
        img[10, 10:-9] = 128
        img[10:-9, -10] = 128
        img[-10, 10:-9] = 128
        cudasift.ExtractKeypoints(img, data)
        self.assertEqual(len(data), 4)
        df, keypoints = data.to_data_frame()
        idx = np.lexsort((df.xpos, df.ypos))
        #
        # Check that the four corners are just inside the square
        #
        for i in (0, 1):
            self.assertTrue(df.ypos[idx[i]] > 10 and df.ypos[idx[i]] < 15)
        for i in (2, 3):
            self.assertTrue(df.ypos[idx[i]] > 85 and df.ypos[idx[i]] < 90)
        for i in (0, 2):
            self.assertTrue(df.xpos[idx[i]] > 10 and df.xpos[idx[i]] < 15)
        for i in (1, 3):
            self.assertTrue(df.xpos[idx[i]] > 85 and df.xpos[idx[i]] < 90)

    def test_01_03_from_data_frame(self):
        data = cudasift.PySiftData(100)
        img = np.zeros((100, 100), np.uint8)
        img[10:-9, 10] = 128
        img[10, 10:-9] = 128
        img[10:-9, -10] = 128
        img[-10, 10:-9] = 128
        cudasift.ExtractKeypoints(img, data)
        self.assertEqual(len(data), 4)
        df, keypoints = data.to_data_frame()
        data2 = cudasift.PySiftData.from_data_frame(df, keypoints)
        df2, keypoints2 = data2.to_data_frame()
        np.testing.assert_array_equal(keypoints, keypoints2)
        np.testing.assert_array_equal(df2.xpos, df.xpos)
        np.testing.assert_array_equal(df.ypos, df2.ypos)
        np.testing.assert_array_equal(df.scale, df2.scale)
        np.testing.assert_array_equal(df.sharpness, df2.sharpness)
        np.testing.assert_array_equal(df.edgeness, df2.edgeness)
        np.testing.assert_array_equal(df.score, df2.score)
        np.testing.assert_array_equal(df.ambiguity, df2.ambiguity)

    def test_01_04_match(self):
        #
        # A 3/4/5 right triangle
        #
        img = np.zeros((100, 100), np.uint8)
        img[10:90, 10] = 255
        img[10, 10:70] = 255
        img[np.arange(90, 9, -1),
            70 - (np.arange(80, -1, -1) * 3 / 4 + .5).astype(int)] = 255
        data1 = cudasift.PySiftData(100)
        cudasift.ExtractKeypoints(img, data1)
        data2 = cudasift.PySiftData(100)
        cudasift.ExtractKeypoints(img.transpose(), data2)
        cudasift.PyMatchSiftData(data1, data2)
        df1, keypoints1 = data1.to_data_frame()
        df2, keypoints2 = data2.to_data_frame()
        pass

if __name__ == "__main__":
    unittest.main()
