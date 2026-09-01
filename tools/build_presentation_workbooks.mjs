import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(process.argv[2]);
if (!root) throw new Error("Usage: node tools/build_presentation_workbooks.mjs <presentation-archive>");
const inputDir = path.join(root, "workbook_inputs");
const tableDir = path.join(root, "tables");
const previewDir = path.join(root, "workbook_previews");
await fs.mkdir(tableDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

function style(workbook, sheetName) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const used = sheet.getUsedRange();
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  used.format.font = { name: "Aptos", size: 10, color: "#243447" };
  used.format.verticalAlignment = "top";
  used.format.wrapText = true;
  used.format.columnWidth = 18;
  used.format.autofitRows();
  used.format.borders = {
    insideHorizontal: { style: "thin", color: "#E2E8F0" },
    bottom: { style: "thin", color: "#CBD5E1" },
  };
  const header = used.getRow(0);
  header.format.fill = "#17365D";
  header.format.font = { name: "Aptos Display", size: 10, bold: true, color: "#FFFFFF" };
  header.format.rowHeight = 30;
  return sheet;
}

async function build(outputFile, specs, previewName) {
  try {
    await fs.access(outputFile);
    throw new Error(`Refusing to overwrite workbook: ${outputFile}`);
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  const workbook = Workbook.create();
  for (const [sheetName, csvName] of specs) {
    const csv = await fs.readFile(path.join(inputDir, csvName), "utf8");
    await workbook.fromCSV(csv, { sheetName });
    style(workbook, sheetName);
  }
  const inspect = await workbook.inspect({ kind: "sheet,region,formula", maxChars: 12000, tableMaxRows: 6, tableMaxCols: 10, options: { maxResults: 120 } });
  const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, summary: "formula error scan" });
  const renders = [];
  for (const [sheetName] of specs) {
    const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 0.8, format: "png" });
    const previewPath = path.join(previewDir, `${previewName}_${sheetName}.png`);
    await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
    renders.push(previewPath);
  }
  const exported = await SpreadsheetFile.exportXlsx(workbook);
  await exported.save(outputFile);
  return { outputFile, sheets: specs.map(([name]) => name), inspect, errors, renders };
}

const master = [
  ["dataset", "dataset.csv"], ["denoising metrics", "denoising_metrics.csv"],
  ["literature comparison", "literature_comparison.csv"], ["debug", "debug.csv"],
  ["E0", "e0.csv"], ["Stage2 ablation", "stage2_ablation.csv"],
  ["postprocessing", "postprocessing.csv"], ["Joint", "joint.csv"],
  ["input experiment", "input_experiment.csv"], ["position results", "position_results.csv"],
  ["seed results", "seed_results.csv"], ["image index", "image_index.csv"],
  ["missing assets", "missing_assets.csv"], ["conclusions", "conclusions.csv"],
  ["limitations", "limitations.csv"],
];
const jobs = [
  [path.join(root, "SABIDS_presentation_results.xlsx"), master, "master"],
  [path.join(tableDir, "stage1_metrics_summary.xlsx"), [["dataset", "dataset.csv"], ["frames", "denoising_metrics.csv"], ["positions", "position_results.csv"]], "stage1"],
  [path.join(tableDir, "literature_denoising_comparison.xlsx"), [["literature", "literature_comparison.csv"], ["dataset protocol", "dataset.csv"]], "literature"],
  [path.join(tableDir, "stage2_ablation.xlsx"), [["ablation", "stage2_ablation.csv"], ["postprocessing", "postprocessing.csv"], ["debug", "debug.csv"]], "stage2"],
  [path.join(tableDir, "joint.xlsx"), [["Joint", "joint.csv"], ["seed results", "seed_results.csv"]], "joint"],
  [path.join(tableDir, "input_experiment.xlsx"), [["input experiment", "input_experiment.csv"]], "input"],
];
const results = [];
for (const [outputFile, specs, previewName] of jobs) results.push(await build(outputFile, specs, previewName));
await fs.writeFile(path.join(root, "workbook_verification.json"), JSON.stringify({ generatedAt: new Date().toISOString(), workbooks: results }, null, 2), "utf8");
console.log(JSON.stringify({ status: "complete", outputs: results.map(result => result.outputFile) }, null, 2));
