"""Deployment gate: an Align-ULCNet graph against the AEC lane that feeds it.

The C model I/O descriptor (``ulcnet_model_io_descriptor_validate``) checks only
the tensor boundary: layout version, grid, D, state shapes and far-input mode.
A graph whose weights were trained on another linear-AEC front end (a retired
``aiaec-linear-error-v3`` corpus, another behaviour hash, another filter
length) has the same tensors and passes it. Two checks close that gap before a
graph is deployed:

* :func:`lane_mismatches` compares the AEC configuration the C lane actually
  builds (``AIAEC/tests/deployed_lane_dump.c`` prints it from the deployed
  ``derive_lane``) with the training contract
  (``dataset_gen.linear_aec.LinearAecContract``).
* :func:`metadata_mismatches` compares the graph's JSON sidecar
  (``export_onnx._write_metadata``) with the compiled
  ``ULCNET_MODEL_IO_FRONTEND_CONTRACT``, the live
  ``aec_python_behavior_hash()`` and the graph and checkpoint bytes.

Both return a list of human-readable mismatches; empty means compatible.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional

from AIAEC.Align_ULCNet.export_onnx import ERROR_INPUT_SEAM, file_sha256
from AIAEC.dataset_gen.linear_aec import (
    DATASET_DELAY_NUM_FILTERS,
    RETIRED_BEHAVIOR_HASHES,
    RETIRED_CONTRACT_VERSIONS,
    LinearAecContract,
    make_linear_aec_config,
)


def parse_lane_dump(text: str) -> Dict[str, str]:
    """``key=value`` lines of deployed_lane_dump into a dict."""
    lane = {}
    for line in text.splitlines():
        if line.strip():
            key, sep, value = line.partition('=')
            if not sep:
                raise ValueError('lane dump line without "=": %r' % line)
            lane[key.strip()] = value.strip()
    return lane


def _int(lane: Mapping[str, str], key: str) -> int:
    return int(lane[key])


def _flag(lane: Mapping[str, str], key: str) -> bool:
    value = _int(lane, key)
    if value not in (0, 1):
        raise ValueError('lane %s=%r is not a 0/1 flag' % (key, lane[key]))
    return bool(value)


def _expect(out: List[str], name: str, actual, expected) -> None:
    if actual != expected:
        out.append('%s: lane %r, contract %r' % (name, actual, expected))


def _cng(lane, contract, cfg, name, out):
    # lib/aec applies comfort noise only in its own output synthesis, which a
    # context lane does not run; the lane inherits enable_cng = 1 from the
    # level factory. Accepted only on the analysis-seam lane, and only with the
    # dump's measurement that formed_hop is byte-identical with CNG off.
    actual = _flag(lane, 'enable_cng')
    if actual == contract.enable_cng:
        return
    inert = (not contract.enable_cng
             and _int(lane, 'spatial_linear_context')
             == _int(lane, 'context_linear_only')
             and _flag(lane, 'formed_hop_cng_invariant'))
    if not inert:
        out.append(
            'enable_cng: lane %r, contract %r (accepted only on a '
            'LINEAR_ONLY lane whose formed_hop is unchanged by it)'
            % (actual, contract.enable_cng))


def _seam(lane, contract, cfg, name, out):
    # formed_output: the selected, crossfaded, WOLA-formed linear residual
    # with RES and CNG off, i.e. AecResContext.formed_hop of a context lane.
    _expect(out, 'output_seam (return_res_context)',
            _flag(lane, 'return_res_context'), True)
    _expect(out, 'output_seam (spatial_linear_context LINEAR_ONLY)',
            _int(lane, 'spatial_linear_context'),
            _int(lane, 'context_linear_only'))


def _preset(lane, contract, cfg, name, out):
    # The level selects the RES floors; compared as values so a level whose
    # floors differ from the preset's is caught even when its name matches.
    _expect(out, 'preset (min_gain_floor_far_active_db)',
            float(lane['min_gain_floor_far_active_db']),
            float(cfg.min_gain_floor_far_active_db))
    _expect(out, 'preset (min_gain_floor_dt_db)',
            float(lane['min_gain_floor_dt_db']),
            float(cfg.min_gain_floor_dt_db))
    _expect(out, 'preset (min_gain_split_floor_enabled)',
            _flag(lane, 'min_gain_split_floor_enabled'),
            bool(cfg.min_gain_split_floor_enabled))


def _delay(lane, contract, cfg, name, out):
    _expect(out, 'enable_delay_est', _flag(lane, 'enable_delay_est'),
            contract.enable_delay_est)
    _expect(out, 'enable_delay_est (delay_mode MATCHED)',
            _int(lane, 'delay_mode'), _int(lane, 'delay_mode_matched'))


def _field(lane_key: str, read: Callable) -> Callable:
    def check(lane, contract, cfg, name, out):
        _expect(out, name, read(lane, lane_key), getattr(contract, name))
    return check


# Every LinearAecContract field, mapped to its lane check or to the reason it
# is not a lane field. tests/test_deployed_frontend_contract.py requires the
# keys to equal the contract's fields, so a new contract field fails the test
# until it is mapped here.
CONTRACT_FIELDS: Dict[str, object] = {
    'version': _field('frontend_contract', lambda lane, key: lane[key]),
    'engine': 'names the Python materializer; the C lane is lib/aec\'s C port '
              'of the same PBFDKF, whose equivalence the C/Python parity '
              'suites test',
    'preset': _preset,
    'sample_rate': _field('sample_rate', _int),
    'frame_size': _field('fft_size', _int),
    'hop_size': _field('hop_size', _int),
    'filter_length': _field('filter_length', _int),
    'mode': 'lib/aec\'s C AEC has the PBFDKF filter only and no mode field',
    'enable_res': _field('enable_res', _flag),
    'enable_cng': _cng,
    'enable_shadow': _field('enable_shadow', _flag),
    'enable_delay_est': _delay,
    'enable_highpass': _field('enable_highpass', _flag),
    'output_seam': _seam,
    'limiter': 'product-level output dynamics after the AEC chain; the stage '
               'reads formed_hop with nothing in between',
    'aec_commit': 'provenance only (not a compatibility field)',
    'aec_source_hash': 'provenance only (not a compatibility field)',
    'aec_behavior_hash': 'graph metadata: metadata_mismatches()',
    'behavior_hash_schema': 'folded into aec_behavior_hash',
}

def lane_mismatches(lane: Mapping[str, str],
                    contract: LinearAecContract) -> List[str]:
    """Differences between a deployed lane dump and the training contract.

    Covers every contract field (see CONTRACT_FIELDS) plus the settings of
    the contract's AecConfig (make_linear_aec_config) that move the formed
    seam but are not recorded in the contract dict: partitions, saturation,
    the delay bank, live observer and the delay analysis rate.
    """
    cfg = make_linear_aec_config(
        contract.sample_rate, preset=contract.preset,
        frame_size=contract.frame_size, filter_length=contract.filter_length,
    )
    out: List[str] = []
    for name, check in CONTRACT_FIELDS.items():
        if not isinstance(check, str):
            check(lane, contract, cfg, name, out)

    hop = contract.hop_size
    _expect(out, 'n_partitions', _int(lane, 'n_partitions'),
            max(1, (contract.filter_length + hop - 1) // hop))
    _expect(out, 'enable_saturation', _flag(lane, 'enable_saturation'),
            bool(cfg.enable_saturation_detect))
    _expect(out, 'saturation_softclip_ref',
            _flag(lane, 'saturation_softclip_ref'),
            bool(cfg.saturation_softclip_ref))
    _expect(out, 'delay_num_filters', _int(lane, 'delay_num_filters'),
            DATASET_DELAY_NUM_FILTERS)
    _expect(out, 'fixed_delay_samples', _int(lane, 'fixed_delay_samples'),
            int(cfg.fixed_delay_samples))
    _expect(out, 'live_observer_enabled',
            _flag(lane, 'live_observer_enabled'),
            bool(cfg.live_observer_enabled))
    # lib/aec's Python delay estimator ignores delay_est_period_s and analyses
    # every hop (modules/delay/legacy_compat.py), so the materialized data has
    # no duty cycle; the C lane must not decimate either.
    _expect(out, 'delay_analysis_every_hop',
            _flag(lane, 'delay_analysis_every_hop'), True)
    return out


def metadata_mismatches(metadata: Mapping[str, object], *,
                        frontend_contract: str,
                        behavior_hash: str,
                        graph_path: str,
                        lane: Optional[Mapping[str, str]] = None,
                        checkpoint_path: Optional[str] = None) -> List[str]:
    """Differences between a graph's JSON sidecar and the deployment.

    ``frontend_contract`` is the compiled ULCNET_MODEL_IO_FRONTEND_CONTRACT,
    ``behavior_hash`` the live aec_python_behavior_hash() of the AEC the lane
    is built from, ``graph_path`` the graph file being deployed. ``lane``
    (a lane dump) adds the grid; ``checkpoint_path`` the checkpoint bytes.
    """
    out: List[str] = []
    version = metadata.get('linear_aec_contract_version')
    if version in RETIRED_CONTRACT_VERSIONS:
        out.append('linear_aec_contract_version %r is retired' % version)
    elif version != frontend_contract:
        out.append('linear_aec_contract_version: graph %r, build %r'
                   % (version, frontend_contract))
    graph_hash = metadata.get('aec_behavior_hash')
    if graph_hash in RETIRED_BEHAVIOR_HASHES:
        out.append('aec_behavior_hash %r is retired' % graph_hash)
    elif graph_hash != behavior_hash:
        out.append('aec_behavior_hash: graph %r, live %r'
                   % (graph_hash, behavior_hash))
    if metadata.get('error_input_seam') != ERROR_INPUT_SEAM:
        out.append('error_input_seam: graph %r, expected %r'
                   % (metadata.get('error_input_seam'), ERROR_INPUT_SEAM))
    recorded = metadata.get('onnx_sha256')
    actual = file_sha256(graph_path)
    if recorded is None:
        out.append('onnx_sha256 missing: the sidecar does not identify a graph')
    elif recorded != actual:
        out.append('onnx_sha256: sidecar %r, graph file %r' % (recorded, actual))
    if checkpoint_path is not None:
        actual = file_sha256(checkpoint_path)
        if metadata.get('checkpoint_sha256') != actual:
            out.append('checkpoint_sha256: sidecar %r, checkpoint file %r'
                       % (metadata.get('checkpoint_sha256'), actual))
    if lane is not None:
        for key, lane_key in (('sample_rate', 'sample_rate'),
                              ('n_fft', 'fft_size'),
                              ('hop_len', 'hop_size')):
            if metadata.get(key) != _int(lane, lane_key):
                out.append('%s: graph %r, lane %r'
                           % (key, metadata.get(key), _int(lane, lane_key)))
    return out
