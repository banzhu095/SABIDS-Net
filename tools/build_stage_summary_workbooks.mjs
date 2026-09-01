import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(process.argv[2]);
if (!root) throw new Error("Usage: node tools/build_stage_summary_workbooks.mjs <stage-summary-dir>");
const inputDir = path.join(root, "workbook_inputs");
const tablesDir = path.join(root, "tables");
const previewDir = path.join(root, "workbook_previews");
await fs.mkdir(tablesDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const masterSheets = [
  ["README", "README.csv"], ["experiment_timeline", "experiment_timeline.csv"],
  ["run_registry", "run_registry.csv"], ["protocol_audit", "protocol_audit.csv"],
  ["stage1_denoising", "stage1_denoising.csv"], ["stage2_ablation", "stage2_ablation.csv"],
  ["postprocessing", "postprocessing.csv"], ["joint_factorial", "joint_factorial.csv"],
  ["input_experiment", "input_experiment.csv"], ["metrics_long", "metrics_long.csv"],
  ["position_results", "position_results.csv"], ["training_trajectory", "training_trajectory.csv"],
  ["image_inventory", "image_inventory.csv"], ["evidence_matrix", "evidence_matrix.csv"],
  ["limitations", "limitations.csv"], ["missing_assets", "missing_assets.csv"],
];

function styleSheet(workbook, sheetName) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const used = sheet.getUsedRange();
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  used.format.font = { name: "Aptos", size: 10, color: "#243447" };
  used.format.wrapText = true;
  used.format.verticalAlignment = "top";
  used.format.borders = { preset: "all", style: "thin", color: "#D8E1E8" };
  used.format.columnWidth = 18;
  used.format.autofitRows();
  const header = used.getRow(0);
  header.format.fill = "#17365D";
  header.format.font = { name: "Aptos Display", size: 10, bold: true, color: "#FFFFFF" };
  header.format.rowHeight = 30;
  return sheet;
}

async function buildWorkbook(sheetSpecs, outputFile, previewName) {
  const workbook = Workbook.create();
  for (const [sheetName, csvName] of sheetSpecs) {
    const text = await fs.readFile(path.join(inputDir, csvName), "utf8");
    await workbook.fromCSV(text, { sheetName });
    styleSheet(workbook, sheetName);
  }
  const inspect = await workbook.inspect({
    kind: "sheet,region,formula",
    maxChars: 18000,
    tableMaxRows: 6,
    tableMaxCols: 10,
    options: { maxResults: 160 },
  });
  const formulaErrors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "final formula error scan",
  });
  const preview = await workbook.render({ sheetName: sheetSpecs[0][0], autoCrop: "all", scale: 0.9, format: "png" });
  await fs.writeFile(path.join(previewDir, `${previewName}.png`), new Uint8Array(await preview.arrayBuffer()));
  const out = await SpreadsheetFile.exportXlsx(workbook);
  await out.save(outputFile);
  return { outputFile, sheets: sheetSpecs.map(([name]) => name), inspect, formulaErrors };
}

const results = [];
results.push(await buildWorkbook(masterSheets, path.join(root, "SABIDS_stage_experiment_ledger.xlsx"), "master_README"));

const tableSpecs = [
  ["table_01_dataset_and_split.xlsx", [["run_registry", "run_registry.csv"], ["protocol_audit", "protocol_audit.csv"], ["checkpoint_inventory", "checkpoint_inventory.csv"]]],
  ["table_02_stage1_denoising.xlsx", [["stage1_denoising", "stage1_denoising.csv"]]],
  ["table_03_stage2_ablation.xlsx", [["stage2_ablation", "stage2_ablation.csv"], ["protocol_audit", "protocol_audit.csv"]]],
  ["table_04_postprocessing.xlsx", [["postprocessing", "postprocessing.csv"], ["position_results", "position_results.csv"]]],
  ["table_05_joint_factorial.xlsx", [["joint_factorial", "joint_factorial.csv"]]],
  ["table_06_input_experiment.xlsx", [["input_experiment", "input_experiment.csv"]]],
  ["table_07_claims_and_limitations.xlsx", [["evidence_matrix", "evidence_matrix.csv"], ["limitations", "limitations.csv"], ["missing_assets", "missing_assets.csv"]]],
];
for (const [fileName, sheets] of tableSpecs) {
  results.push(await buildWorkbook(sheets, path.join(tablesDir, fileName), fileName.replace(".xlsx", "")));
}

await fs.writeFile(path.join(root, "workbook_verification.json"), JSON.stringify({ generatedAt: new Date().toISOString(), workbooks: results }, null, 2), "utf8");
console.log(JSON.stringify({ status: "complete", outputs: results.map(r => r.outputFile) }, null, 2));
