from pathlib import Path


AINR = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (AINR / relative).read_text(encoding="utf-8")


def test_every_ainr_trainer_rejects_nonfinite_gradients():
    for relative in (
        "RNNoise-ERB/train.py",
        "GTCRN/train.py",
        "DeepFilterNet2/train.py",
        "DeepFilterNet3/train.py",
    ):
        source = _source(relative)
        assert "error_if_nonfinite=True" in source, relative
        assert "non-finite training loss" in source or "loss is non-finite" in source


def test_cosine_trainers_do_not_restore_stale_tmax():
    for relative in (
        "RNNoise-ERB/train.py",
        "DeepFilterNet2/train.py",
        "DeepFilterNet3/train.py",
    ):
        source = _source(relative)
        assert "scheduler.load_state_dict" not in source, relative
        assert "'global_step': global_step" in source, relative
        assert "for _ in range(global_step)" in source, relative
