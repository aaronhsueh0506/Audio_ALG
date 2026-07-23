"""Python contract tests for log_erb_dfn_mean_cplx_unit_0_4k_v3."""

import unittest
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    FEATURE_VERSION,
    RNNoiseModel,
    compute_erb_matrix,
    erb_bandborder,
    extract_model_features,
    make_norm_alpha,
    normalize_complex_spectrum,
    normalize_log_erb,
    require_checkpoint_feature_config,
)


class DualFeatureTest(unittest.TestCase):
    def setUp(self):
        self.alpha = make_norm_alpha(16000, 256, 1.0)
        self.erb_kwargs = dict(
            norm_alpha=self.alpha,
            init_lo_db=-60.0,
            init_hi_db=-90.0,
            scale_db=40.0,
            clip=5.0,
        )
        self.spec_kwargs = dict(
            norm_alpha=self.alpha,
            init_lo=0.001,
            init_hi=0.0001,
            eps=1e-12,
            clip=10.0,
        )

    def feature_cfg(self):
        return dict(
            erb_alpha=self.alpha,
            erb_norm_init_lo_db=-60.0,
            erb_norm_init_hi_db=-90.0,
            erb_norm_scale_db=40.0,
            erb_norm_clip=5.0,
            spec_bins=129,
            spec_alpha=self.alpha,
            spec_norm_init_lo=0.001,
            spec_norm_init_hi=0.0001,
            spec_norm_eps=1e-12,
            spec_clip=10.0,
        )

    def checkpoint_cfg(self):
        return dict(
            sr=16000,
            n_fft=512,
            win_len=512,
            hop_len=256,
            lookahead_frames=1,
            n_bands=22,
            erb_tau_sec=1.0,
            erb_alpha=self.alpha,
            erb_norm_init_lo_db=-60.0,
            erb_norm_init_hi_db=-90.0,
            erb_norm_scale_db=40.0,
            erb_norm_clip=5.0,
            spec_max_hz=4000,
            spec_bins=129,
            spec_tau_sec=1.0,
            spec_alpha=self.alpha,
            spec_norm_init_lo=0.001,
            spec_norm_init_hi=0.0001,
            spec_norm_eps=1e-12,
            spec_clip=10.0,
        )

    def test_erb_first_frame_updates_then_subtracts(self):
        x = torch.tensor([[[-72.0, -84.0]]])
        feature, state = normalize_log_erb(x, **self.erb_kwargs)
        init = torch.tensor([-60.0, -90.0]).view(1, 1, 2)
        expected_state = self.alpha * init + (1.0 - self.alpha) * x
        expected = (x - expected_state) / 40.0
        torch.testing.assert_close(state, expected_state)
        torch.testing.assert_close(feature, expected)

    def test_erb_chunking_is_state_equivalent(self):
        generator = torch.Generator().manual_seed(5)
        x = -75.0 + 12.0 * torch.randn(2, 500, 22, generator=generator)
        whole, whole_state = normalize_log_erb(x, **self.erb_kwargs)
        first, state = normalize_log_erb(x[:, :173], **self.erb_kwargs)
        second, state = normalize_log_erb(
            x[:, 173:], norm_state=state, **self.erb_kwargs)
        torch.testing.assert_close(torch.cat([first, second], dim=1), whole)
        torch.testing.assert_close(state, whole_state)

    def test_stationary_erb_mean_norm_converges_to_zero(self):
        envelope = torch.linspace(-94.0, -52.0, 22).view(1, 1, -1)
        x = envelope.expand(1, 4096, -1)
        feature, state = normalize_log_erb(x, **self.erb_kwargs)
        self.assertLess(float(feature[0, -1].abs().max()), 2e-5)
        # Float32 EMA reaches a small quantisation floor after thousands of
        # updates; the emitted feature above is the behavioral contract.
        torch.testing.assert_close(state, envelope, atol=1e-3, rtol=0.0)

    def test_complex_first_frame_uses_per_bin_magnitude(self):
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

    def test_complex_path_retains_partial_level(self):
        re = torch.linspace(0.001, 0.02, 129).view(1, 1, -1)
        im = torch.linspace(-0.005, 0.005, 129).view(1, 1, -1)
        x = torch.complex(re, im).expand(1, 4096, -1)
        low, _ = normalize_complex_spectrum(x, **self.spec_kwargs)
        high, _ = normalize_complex_spectrum(4.0 * x, **self.spec_kwargs)
        ratio = high[0, -1].abs().mean() / low[0, -1].abs().mean()
        self.assertAlmostEqual(float(ratio), 2.0, delta=0.01)

    def test_feature_state_chunking_and_model_shapes(self):
        feature_cfg = self.feature_cfg()
        border = erb_bandborder(22, 16000, 512)
        erb = torch.from_numpy(compute_erb_matrix(border, 512, mode=0))
        rng = np.random.default_rng(3)
        array = rng.normal(size=(2, 257, 17)) + 1j * rng.normal(size=(2, 257, 17))
        spec = torch.from_numpy(array.astype(np.complex64)) * 0.01
        erb_f, spec_f, state, _ = extract_model_features(spec, erb, feature_cfg)
        self.assertEqual(erb_f.shape, (2, 17, 22))
        self.assertEqual(spec_f.shape, (2, 17, 2, 129))
        self.assertEqual(state['erb'].shape, (2, 1, 22))
        self.assertEqual(state['spec'].shape, (2, 1, 129))

        first_erb, first_spec, state, _ = extract_model_features(
            spec[:, :, :7], erb, feature_cfg)
        next_erb, next_spec, state, _ = extract_model_features(
            spec[:, :, 7:], erb, feature_cfg, norm_state=state)
        torch.testing.assert_close(torch.cat([first_erb, next_erb], dim=1), erb_f)
        torch.testing.assert_close(torch.cat([first_spec, next_spec], dim=1), spec_f)

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
        with self.assertRaisesRegex(ValueError, 'retrain'):
            require_checkpoint_feature_config({'state_dict': {}}, self.checkpoint_cfg())

    def test_win_len_mismatch_is_rejected(self):
        feature_cfg = self.checkpoint_cfg()
        saved = {
            'feature_version': FEATURE_VERSION,
            'config': {
                'sr': 16000,
                'n_fft': 512,
                'win_len': 320,
                'hop_len': 256,
                'lookahead_frames': 1,
                'n_bands': 22,
                'feature_version': FEATURE_VERSION,
                'feature_erb_norm_tau_sec': 1.0,
                'feature_erb_norm_alpha': self.alpha,
                'feature_erb_norm_init_lo_db': -60.0,
                'feature_erb_norm_init_hi_db': -90.0,
                'feature_erb_norm_scale_db': 40.0,
                'feature_erb_norm_clip': 5.0,
                'feature_spec_max_hz': 4000,
                'feature_spec_bins': 129,
                'feature_spec_norm_tau_sec': 1.0,
                'feature_spec_norm_alpha': self.alpha,
                'feature_spec_norm_init_lo': 0.001,
                'feature_spec_norm_init_hi': 0.0001,
                'feature_spec_norm_eps': 1e-12,
                'feature_spec_clip': 10.0,
            },
        }
        with self.assertRaisesRegex(ValueError, 'win_len'):
            require_checkpoint_feature_config(saved, feature_cfg)


if __name__ == '__main__':
    print(f'feature version: {FEATURE_VERSION}')
    unittest.main()
