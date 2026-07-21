"""Python contract tests for log_erb_shared_online_cmvn_v1.

Run inside the RNNoise training environment:
    make test-feature-python
"""

import math
import unittest

import torch

from train import FEATURE_VERSION, make_norm_alpha, normalize_log_erb


class SharedOnlineNormTest(unittest.TestCase):
    def setUp(self):
        self.alpha = make_norm_alpha(16000, 256, 10.0)
        self.kwargs = dict(
            norm_alpha=self.alpha,
            mean_init_db=-75.0,
            var_init_db2=400.0,
            var_floor_db2=36.0,
            clip=5.0,
        )

    def test_first_frame_uses_previous_state(self):
        # Keep the band average different from mean_init so this also proves
        # that the state update happens after the feature is emitted.
        x = torch.tensor([[[-95.0, -75.0, -54.0]]])
        feature, state = normalize_log_erb(x, **self.kwargs)
        expected = (x[0, 0] + 75.0) / math.sqrt(400.0 + 36.0)
        torch.testing.assert_close(feature[0, 0], expected)
        self.assertNotEqual(float(state['mean'][0, 0]), -75.0)

    def test_chunking_is_state_equivalent(self):
        t = torch.arange(600, dtype=torch.float32).view(1, -1, 1)
        bands = torch.linspace(-92.0, -48.0, 22).view(1, 1, -1)
        x = bands + 4.0 * torch.sin(t / 19.0)

        whole, whole_state = normalize_log_erb(x, **self.kwargs)
        first, state = normalize_log_erb(x[:, :173], **self.kwargs)
        second, state = normalize_log_erb(x[:, 173:], norm_state=state, **self.kwargs)

        torch.testing.assert_close(torch.cat([first, second], dim=1), whole)
        torch.testing.assert_close(state['mean'], whole_state['mean'])
        torch.testing.assert_close(state['var'], whole_state['var'])

    def test_stationary_spectral_envelope_does_not_collapse(self):
        spectrum = torch.linspace(-94.0, -52.0, 22).view(1, 1, -1)
        x = spectrum.expand(1, 4096, -1)
        feature, state = normalize_log_erb(x, **self.kwargs)
        last = feature[0, -1]

        self.assertGreater(float(last.max() - last.min()), 0.25)
        self.assertLess(float(state['var'][0, 0]), 1.0)
        self.assertAlmostEqual(float(state['mean'][0, 0]), float(spectrum.mean()), places=2)

    def test_state_is_one_scalar_pair_per_stream(self):
        x = torch.zeros(3, 8, 22)
        _, state = normalize_log_erb(x, **self.kwargs)
        self.assertEqual(state['mean'].shape, (3, 1))
        self.assertEqual(state['var'].shape, (3, 1))


if __name__ == '__main__':
    print(f'feature version: {FEATURE_VERSION}')
    unittest.main()
