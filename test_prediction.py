"""Tests of frame alignment, CU interpretation, and feature leakage guards."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from prediction_features import read_encoder_csv, feature_matrix, feature_groups
from predict_psnr import grouped_splits, predict_portable


class EncoderFeatureTests(unittest.TestCase):
    def write_csv(self, rows):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / 'encoder.csv'
        path.write_text('Encode Order,Type,POC,QP,Bits,Intra 64x64 DC,4x4,Inter 64x64,Skip 64x64,Merge 64x64,Avg Luma Distortion,Avg Luma Level\n' + '\n'.join(rows))
        return path

    def test_closed_gop_poc_reset_and_b_display_order(self):
        path = self.write_csv(['0,I-SLICE,0,20,800,100%,0%,0%,0%,0%,999,100',
                              '1,P-SLICE,2,22,80,10%,0%,30%,5%,55%,999,101',
                              '2,b-SLICE,1,24,40,0%,0%,30%,10%,60%,999,102',
                              '3,I-SLICE,0,21,800,100%,0%,0%,0%,0%,999,103'])
        result = read_encoder_csv(path, gop=3)
        self.assertEqual(result.frame.tolist(), [0, 1, 2, 3])
        self.assertEqual(result.pict_type.tolist(), ['I', 'B', 'P', 'I'])
        self.assertEqual(result.qp.tolist(), [20, 24, 22, 21])
        self.assertAlmostEqual(result.loc[2, 'intra_cu_pct'], 10)
        self.assertAlmostEqual(result.loc[2, 'inter_cu_pct'], 90)
        self.assertAlmostEqual(result.loc[2, 'skip_merge_cu_pct'], 60)
        self.assertNotIn('Avg Luma Distortion', result.columns)

    def test_missing_or_duplicate_frame_is_rejected(self):
        path = self.write_csv(['0,I-SLICE,0,20,800,100%,0%,0%,0%,0%,999,100',
                              '1,P-SLICE,0,22,80,10%,0%,30%,5%,55%,999,101'])
        with self.assertRaises(ValueError):
            read_encoder_csv(path)

    def test_cu_percentage_not_covering_whole_frame_is_rejected(self):
        path = self.write_csv(['0,I-SLICE,0,20,800,20%,0%,0%,0%,0%,999,100'])
        with self.assertRaises(ValueError):
            read_encoder_csv(path)

    def test_labels_and_video_identity_never_enter_features(self):
        raw = pd.DataFrame({'sequence':['a','b'], 'target_kbps':[150,300],
            'pict_type':['I','P'], 'frame':[0,1], 'si_2023':[40,50], 'ti_2023':[np.nan,3],
            'qp':[20,30], 'intra_cu_pct':[100,10], 'inter_cu_pct':[0,90],
            'skip_merge_cu_pct':[0,50], 'packet_bytes':[100,200],
            'source_luma_mean':[100,120], 'source_luma_std':[20,30],
            'mse_y':[1,100], 'psnr_y':[48,28], 'sha256_encoded':['secret','secret']})
        altered = raw.copy()
        altered['psnr_y'] = -999
        altered['mse_y'] = -999
        altered['sequence'] = 'changed'
        for columns in feature_groups('frame').values():
            pd.testing.assert_frame_equal(feature_matrix(raw, 'frame')[columns],
                                          feature_matrix(altered, 'frame')[columns])

    def test_all_rates_of_a_source_stay_in_same_outer_and_inner_fold(self):
        groups = np.repeat(['a', 'b', 'c', 'd', 'e'], 6)
        covered = []
        for train, test in grouped_splits(groups):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            self.assertEqual(len(set(groups[test])), 1)
            covered.extend(test)
            for inner_train, inner_test in grouped_splits(groups[train], n_splits=3):
                self.assertFalse(set(groups[train][inner_train]) & set(groups[train][inner_test]))
        self.assertEqual(sorted(covered), list(range(len(groups))))

    def test_portable_model_handles_missing_value_and_indicator(self):
        model = {'input_columns':['x'], 'median':[10.], 'missing_indicator_indices':[0],
                 'scale_mean':[10., 0.], 'scale_std':[2., 1.],
                 'coefficient':[3., 5.], 'intercept':20.}
        result = predict_portable(pd.DataFrame({'x':[12., np.nan]}), model)
        np.testing.assert_allclose(result, [23., 25.])

    def test_primary_inference_does_not_require_unused_features_or_labels(self):
        raw = pd.DataFrame({'target_kbps':[300], 'pict_type':['P'], 'frame':[1],
            'si_2023':[50.], 'ti_2023':[3.], 'qp':[25.],
            'intra_cu_pct':[10.], 'skip_merge_cu_pct':[60.]})
        columns = feature_groups('frame')['si_ti_qp_cu']
        result = feature_matrix(raw, 'frame', columns=columns)
        self.assertEqual(result.columns.tolist(), columns)
        self.assertTrue(np.isfinite(result.to_numpy()).all())


if __name__ == '__main__':
    unittest.main()
