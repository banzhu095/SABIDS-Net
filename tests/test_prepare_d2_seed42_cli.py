from tools.prepare_d2_seed42 import _arguments


def test_prepare_d2_seed42_accepts_explicit_device_without_changing_defaults():
    required = [
        "--mode", "overfit",
        "--arms", "D20",
        "--tag", "fixture",
        "--d1-checkpoint", "d1.pth",
        "--d1-initial-inventory", "initial.json",
        "--d1-checkpoint-binding", "binding.json",
        "--protocol-lock", "lock.json",
        "--split-contract", "split.yaml",
        "--vessel-strata-definition", "strata.json",
    ]
    assert _arguments(required).device is None
    assert _arguments([*required, "--device", "cuda"]).device == "cuda"
