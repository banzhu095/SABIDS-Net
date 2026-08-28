from tools.compare_stage2_protocols import (
    FINGERPRINT_FIELDS,
    HASH_SCHEMA_VERSION,
    _field_comparison,
    _label_differences,
    _protocol_comparison,
)


def complete_metadata():
    result = {field: f"hash-{field}" for field in FINGERPRINT_FIELDS}
    result.update({"hash_schema_version": HASH_SCHEMA_VERSION, "metadata_version": 2,
                   "missing_label_assets": []})
    return result


def test_complete_self_reference_is_matched_but_legacy_is_unknown():
    current = complete_metadata()
    assert _field_comparison(current, current)["status"] == "matched"
    legacy = {"manifest_sha256": "old"}
    assert _field_comparison(legacy, legacy)["status"] == "unknown"


def test_label_audit_distinguishes_serialization_from_decoded_change():
    reference = {"g|layer|0": {"raw_sha256": "raw-a", "decoded_sha256": "pixels-a"}}
    serialized = {"g|layer|0": {"raw_sha256": "raw-b", "decoded_sha256": "pixels-a"}}
    changed = {"g|layer|0": {"raw_sha256": "raw-c", "decoded_sha256": "pixels-c"}}
    assert _label_differences(reference, serialized)["differences"][0]["status"] == "serialization_only"
    assert _label_differences(reference, changed)["differences"][0]["status"] == "decoded_content_different"


def test_protocol_requires_exact_declared_difference():
    reference = {"loss.weights.outside": 0.5, "train.epochs": 60}
    candidate = {"loss.weights.outside": 0.0, "train.epochs": 60}
    assert _protocol_comparison(reference, candidate, ["loss.weights.outside"])["status"] == "matched"
    assert _protocol_comparison(reference, candidate, ["train.epochs"])["status"] == "different"
