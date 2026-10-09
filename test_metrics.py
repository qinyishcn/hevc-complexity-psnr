import unittest
import numpy as np
from metrics import spatial_information, temporal_information, psnr_from_mse, pq_luma


class MetricTests(unittest.TestCase):
    def test_static_and_uniform_change(self):
        y = np.arange(64, dtype=np.float64).reshape(8, 8)
        self.assertEqual(temporal_information(y, y), 0)
        self.assertEqual(temporal_information(y + 10, y), 0)

    def test_signed_difference_no_wraparound(self):
        before = np.array([[250, 0], [0, 250]], dtype=np.uint8)
        after = np.array([[0, 250], [250, 0]], dtype=np.uint8)
        self.assertEqual(temporal_information(after, before), 250)

    def test_constant_and_linear_ramp_spatial_information(self):
        self.assertEqual(spatial_information(np.ones((10, 10)) * 100), 0)
        ramp = np.tile(np.arange(10, dtype=np.float64), (10, 1))
        self.assertEqual(spatial_information(ramp), 0)

    def test_known_psnr(self):
        self.assertAlmostEqual(psnr_from_mse(100), 28.130803608679106)
        self.assertEqual(psnr_from_mse(0), float('inf'))

    def test_sobel_edge_fixture(self):
        y = np.zeros((5, 5)); y[:, 3:] = 10
        # Valid 3x3 region: each row has magnitudes [0, 40, 40].
        expected = np.std([0, 40, 40] * 3)
        self.assertAlmostEqual(spatial_information(y), expected)

    def test_p910_2023_sdr_endpoints_and_clipping(self):
        result = pq_luma(np.array([0, 16, 235, 255], dtype=np.uint8))
        self.assertAlmostEqual(result[1] / 255, 0.02148621379868528, places=10)
        self.assertAlmostEqual(result[2] / 255, 0.6218628370226069, places=10)
        self.assertEqual(result[0], result[1])
        self.assertEqual(result[2], result[3])
        self.assertTrue((np.diff(pq_luma(np.arange(256))) >= 0).all())

    def test_full_range_sensitivity_does_not_clip_superwhites(self):
        result = pq_luma(np.array([0, 16, 235, 255]), color_range='full')
        self.assertTrue((np.diff(result)>0).all())
        self.assertAlmostEqual(result[0]/255,0.02148621379868528,places=10)
        self.assertAlmostEqual(result[-1]/255,0.6218628370226069,places=10)


if __name__ == '__main__':
    unittest.main()
