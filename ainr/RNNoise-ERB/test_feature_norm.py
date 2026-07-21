"""Python contract tests for log_erb_abs_cplx_0_4k_v2."""

import unittest

import numpy as np
import torch

from train import (
    FEATURE_VERSION,
    RNNoiseModel,
    compute_erb_matrix,
    erb_bandborder,
    extract_model_features,
    make_norm_alpha,
    normalize_absolute_log_erb,
    normalize_complex_spectrum,
    require_checkpoint_feature_config,
)


class DualFeatureTest(unittest.TestCase):
    def setUp(self):
        self.alpha = make_norm_alpha(16000, 256, 1.0)
        self.spec_kwargs = dict(
            norm_alpha=self.alpha,
            init_lo=0.001,
            init_hi=0.0001,
            eps=1e-12,
            clip=10.0,
        )

    def test_absolute_erb_has_no_temporal_collapse(self):
        envelope = torch.linspace(-94.0, -52.0, 22).view(1, 1, -1)
        x = envelope.expand(1, 4096, -1)
        feature = normalize_absolute_log_erb(x, -75.0, 20.0, 5.0)
        torch.testing.assert_close(feature[:, 0], feature[:, -1])
        self.assertGreater(float(feature[0, -1].max() - feature[0, -1].min()), 2.0)

    def test_absolute_erb_keeps_level(self):
        low = normalize_absolute_log_erb(torch.tensor([-75.0]), -75.0, 20.0, 5.0)
        high = normalize_absolute_log_erb(torch.tensor([-55.0]), -75.0, 20.0, 5.0)
        torch.testing.assert_close(high - low, torch.ones(1))

    def test_complex_first_frame_updates_then_normalizes(self):
        x = torch.complex(torch.tensor([[[0.04, 0.01]]]),
                          torch.tensor([[[0.03, -0.02]]]))
        feature, state = normalize_complex_spectrum(x, **self.spec_kwargs)
        init = torch.tensor([0.001, 0.0001]).view(1, 1, 2)
        expected_state = self.alpha * init + (1.0 - self.alpha) * x.abs()
        expected = torch.view_as_real(x / torch.sqrt(expected_state + 1e-12))
        expected = expected.permute(0, 1, 3, 2)
        torch.testing.assert_close(state, expected_state)
        torch.testing.assert_close(feature, expected)

    def test_complex_chunking_is_state_equivalent(self):
        generator = torch.Generator().manual_seed(7)
        real = torch.randn(2, 500, 129, generator=generator) * 0.01
        imag = torch.randn(2, 500, 129, generator=generator) * 0.01
        x = torch.complex(real, imag)
        whole, whole_state = normalize_complex_spectrum(x, **self.spec_kwargs)
        first, state = normalize_complex_spectrum(x[:, :173], **self.spec_kwargs)
        second, state = normalize_complex_spectrum(
            x[:, 173:], norm_state=state, **self.spec_kwargs)
        torch.testing.assert_close(torch.cat([first, second], dim=1), whole)
        torch.testing.assert_close(state, whole_state)

    def test_stationary_complex_feature_remains_nonzero(self):
        re = torch.linspace(0.001, 0.02, 129).view(1, 1, -1)
        im = torch.linspace(-0.005, 0.005, 129).view(1, 1, -1)
        x = torch.complex(re, im).expand(1, 4096, -1)
        feature, state = normalize_complex_spectrum(x, **self.spec_kwargs)
        self.assertGreater(float(feature[0, -1].abs().mean()), 0.01)
        self.assertEqual(state.shape, (1, 1, 129))

    def test_feature_and_model_shapes(self):
        feature_cfg = dict(
            erb_center_db=-75.0, erb_scale_db=20.0, erb_clip=5.0,
            spec_bins=129, spec_alpha=self.alpha,
            spec_norm_init_lo=0.001, spec_norm_init_hi=0.0001,
            spec_norm_eps=1e-12, spec_clip=10.0,
        )
        border = erb_bandborder(22, 16000, 512)
        erb = torch.from_numpy(compute_erb_matrix(border, 512, mode=0))
        rng = np.random.default_rng(3)
        array = rng.normal(size=(2, 257, 17)) + 1j * rng.normal(size=(2, 257, 17))
        spec = torch.from_numpy(array.astype(np.complex64)) * 0.01
        erb_f, spec_f, state, _ = extract_model_features(spec, erb, feature_cfg)
        self.assertEqual(erb_f.shape, (2, 17, 22))
        self.assertEqual(spec_f.shape, (2, 17, 2, 129))
        self.assertEqual(state.shape, (2, 1, 129))

        model = RNNoiseModel(22, 129, cond_size=16, gru_size=24,
                             spec_conv_channels=4, spec_embed_size=12)
        gain, states = model(erb_f, spec_f)
        self.assertEqual(gain.shape, (2, 15, 22))
        self.assertEqual([x.shape for x in states], [(1, 2, 24)] * 3)
        self.assertTrue(torch.isfinite(gain).all())

        stream_state = None
        stream_gain = []
        for index in range(2, erb_f.shape[1]):
            one_gain, stream_state = model(
                erb_f[:, index - 2:index + 1],
                spec_f[:, index - 2:index + 1], stream_state)
            stream_gain.append(one_gain)
        torch.testing.assert_close(torch.cat(stream_gain, dim=1), gain)

    def test_legacy_checkpoint_is_rejected(self):
        cfg = dict(
            sr=16000, n_fft=512, hop_len=256, lookahead_frames=1,
            erb_center_db=-75.0, erb_scale_db=20.0, erb_clip=5.0,
            spec_max_hz=4000, spec_bins=129, spec_tau_sec=1.0,
            spec_alpha=self.alpha, spec_norm_init_lo=0.001,
            spec_norm_init_hi=0.0001, spec_norm_eps=1e-12, spec_clip=10.0,
        )
        with self.assertRaisesRegex(ValueError, 'retrain'):
            require_checkpoint_feature_config({'state_dict': {}}, cfg)


if __name__ == '__main__':
    print(f'feature version: {FEATURE_VERSION}')
    unittest.main()
