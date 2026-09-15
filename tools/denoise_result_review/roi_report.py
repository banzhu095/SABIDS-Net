from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.oct_denoise_benchmark.data import load_protocol_manifest

from .method_literature import implementation_summary, load_literature
from .workbook import highlight_group_best, write_workbook


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.is_file() else pd.DataFrame()


def _tissue(position: pd.DataFrame, tissue: str) -> pd.DataFrame:
    return position[position.tissue.astype(str) == tissue].copy() if not position.empty and "tissue" in position else pd.DataFrame()


def build_roi_report(project_root: str | Path, run_dir: str | Path, output_root: str | Path) -> dict[str, str]:
    root, run, output = Path(project_root).resolve(), Path(run_dir).resolve(), Path(output_root).resolve()
    metrics, audit = output / "metrics", output / "audit"
    registry = _csv(output / "roi_registry.csv"); per_roi = _csv(metrics / "per_roi_metrics.csv")
    sample = _csv(metrics / "per_sample_tissue_metrics.csv"); position = _csv(metrics / "per_position_tissue_metrics.csv")
    differences = _csv(metrics / "roi_method_differences.csv"); bootstrap = _csv(metrics / "roi_bootstrap_confidence_intervals.csv")
    failures = _csv(metrics / "roi_failures.csv"); cnr = _csv(metrics / "vessel_stroma_cnr.csv")
    inventory = _csv(audit / "local_image_inventory.csv")
    global_metrics = _csv(run / "metrics" / "per_dataset_metrics.csv")
    configuration = implementation_summary(root, run)
    literature = load_literature()
    def global_part(dataset: str, split: str | None = None) -> pd.DataFrame:
        if global_metrics.empty: return pd.DataFrame()
        value = global_metrics[global_metrics.dataset.astype(str) == dataset]
        return value[value.split.astype(str) == split] if split and "split" in value else value
    selected_positions = registry.position_id.nunique() if not registry.empty else 0
    package_positions = inventory.position_id.nunique() if not inventory.empty else 0
    protocol = load_protocol_manifest(root)
    expected_test_positions = protocol[(protocol.dataset.astype(str) == "PKU37") & (protocol.split.astype(str) == "test")].position_id.nunique()
    formal_registry = (
        not registry.empty
        and registry.dataset.astype(str).eq("PKU37").all()
        and registry.split.astype(str).eq("test").all()
    )
    complete_formal_positions = formal_registry and package_positions == expected_test_positions and selected_positions == package_positions
    scope = "fixed_roi_confirmatory_full_test_positions" if complete_formal_positions else "exploratory_descriptive"
    full_image_scope = "none"
    if not global_metrics.empty and {"dataset", "split"}.issubset(global_metrics.columns):
        full_image_scope = "; ".join(sorted(global_metrics[["dataset", "split"]].drop_duplicates().astype(str).agg("/".join, axis=1)))
    full_image_note = f"Available full-image metric scopes: {full_image_scope}. These are not replaced by ROI statistics."
    workbook_path = output / "roi_summary.xlsx"
    sheets = {
        "Method Literature": literature, "Method Configuration": configuration,
        "Global PKU Test": global_part("PKU37", "test"), "Duke17 External": global_part("Duke17"), "Duke28 External": global_part("Duke28"),
        "ROI Registry": registry, "Per ROI Metrics": per_roi, "Vitreous Summary": _tissue(position, "vitreous"),
        "Retina Summary": _tissue(position, "retina"), "Choroid Vessel": _tissue(position, "choroid_vessel"),
        "Choroid Stroma": _tissue(position, "choroid_stroma"), "Vessel-Stroma CNR": cnr,
        "Method Differences": differences, "Bootstrap CI": bootstrap, "Failures": failures, "Image Inventory": inventory,
    }
    write_workbook(workbook_path, sheets, readme=[["ROI denoising summary", ""], ["Source run", str(run)], ["Full-image metric scope", full_image_note], ["ROI scope", scope], ["Independent unit", "Anatomical position; repeated ROIs, frames, and model seeds are not independent cases."], ["Image scaling", "uint8/255, uint16/65535, float/manifest data_range; no per-image normalization."], ["CNR interpretation", "Low-reflectance choroidal vessels are normally darker than surrounding stroma."], ["Published results", "Bibliographic metadata are separate from harmonized local metrics."]])
    highlight_group_best(workbook_path, ["Per ROI Metrics", "Vitreous Summary", "Retina Summary", "Choroid Vessel", "Choroid Stroma"])
    report = output / "roi_analysis_report.md"
    report.write_text(f"# ROI denoising analysis\n\n## Scope\n\n- {full_image_note}\n- ROI statistics: `{scope}` across {selected_positions} selected of {package_positions} packaged positions.\n- Qualitative panels: generated only after ROI locking.\n- No p-values are reported. Position-level bootstrap is emitted only when every packaged test position is covered.\n\n## Interpretation\n\nROI findings do not replace the complete test result. Multiple ROIs, repeated frames, and three model seeds are not counted as independent cases. Choroidal vessel regions are normally darker than stroma; CNR polarity is recorded explicitly.\n\n## Missing assets\n\n{failures.to_markdown(index=False) if not failures.empty else 'No ROI evaluation failures were recorded.'}\n", encoding="utf-8")
    return {"workbook": str(workbook_path), "report": str(report), "scope": scope}
