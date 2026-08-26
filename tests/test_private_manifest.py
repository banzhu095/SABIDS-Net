from tools.prepare_private12x12_data import parse_private_filename, split_rows


def test_parse_private12x12_filename():
    parsed = parse_private_filename(
        "中度NPDR_DYN00402-OS4_sn3347_0048_DYN00402-OS4_sn3347_0125"
    )
    assert parsed["disease"] == "中度NPDR"
    assert parsed["patient_id"] == "DYN00402"
    assert parsed["eye"] == "OS4"
    assert parsed["scan_id"] == "sn3347"
    assert parsed["source_frame_index"] == 48
    assert parsed["frame_index"] == 125
    assert parsed["group_id"] == parsed["sample_id"]


def test_patient_split_has_no_leakage_and_distributes_vessel_labels():
    rows = []
    for patient_index in range(12):
        patient_id = f"DYN{patient_index:05d}"
        for frame in range(2):
            rows.append(
                {
                    "patient_id": patient_id,
                    "scan_id": f"sn{patient_index:04d}",
                    "disease": "PDR" if patient_index % 2 else "NO_DR",
                    "vessel_mask_path": (
                        f"vessel_{patient_index}_{frame}.png"
                        if patient_index in {0, 1, 2}
                        else ""
                    ),
                }
            )
    assignment = split_rows(
        rows,
        split_unit="patient",
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42,
        candidates=512,
    )
    assert set(assignment.values()) == {"train", "val", "test"}
    vessel_splits = {assignment[f"DYN{index:05d}"] for index in (0, 1, 2)}
    assert vessel_splits == {"train", "val", "test"}
