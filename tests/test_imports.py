"""No-GPU sanity check. Catches packaging mistakes before the cluster does."""


def test_models_base_imports():
    from src.models import base

    assert callable(base.load_qwen2vl_with_lora)


def test_utils_seed_imports():
    from src.utils.seed import set_seed

    set_seed(0)
