import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const inputDir = path.resolve(process.argv[2]);
const outputFile = path.resolve(process.argv[3] ?? path.join(inputDir, "experiment_record.xlsx"));
const previewDir = path.join(inputDir, "workbook_previews");

const sheets = [
  ["README", "workbook_readme.csv"],
  ["Runs", "experiment_registry.csv"],
  ["Protocols", "protocol_comparison.csv"],
  ["Methods", "methods_table.csv"],
  ["Training_Best", "training_best.csv"],
  ["Denoising_Val", "denoising_val.csv"],
  ["Layer_Val", "layer_val.csv"],
  ["Vessel_Val", "vessel_val.csv"],
  ["Postprocess", "postprocess_comparison.csv"],
  ["Groups", "groups_table.csv"],
  ["Thresholds", "threshold_comparison.csv"],
  ["Qualitative_Index", "qualitative_index.csv"],
  ["Evidence", "evidence_table.csv"],
  ["Missing", "missing_table.csv"],
];

const workbook = Workbook.create();
for (const [sheetName, csvName] of sheets) {
  const csvText = await fs.readFile(path.join(inputDir, csvName), "utf8");
  await workbook.fromCSV(csvText, { sheetName });
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
}

for (const name of ["README", "Methods", "Evidence", "Missing"]) {
  const sheet = workbook.worksheets.getItem(name);
  sheet.getRange("A:C").format.columnWidth = 28;
  sheet.getRange("C:C").format.columnWidth = 56;
}
workbook.worksheets.getItem("Qualitative_Index").getRange("A:Q").format.columnWidth = 22;
workbook.worksheets.getItem("Qualitative_Index").getRange("Q:Q").format.columnWidth = 60;

const summary = await workbook.inspect({
  kind: "sheet,region,formula",
  maxChars: 12000,
  tableMaxRows: 5,
  tableMaxCols: 8,
  options: { maxResults: 100 },
});
await fs.writeFile(
  path.join(inputDir, "workbook_verification.json"),
  JSON.stringify({ generated_at: new Date().toISOString(), inspect: summary }, null, 2),
  "utf8",
);

await fs.mkdir(previewDir, { recursive: true });
for (const [sheetName] of sheets) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 0.8, format: "png" });
  await fs.writeFile(path.join(previewDir, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputFile);
console.log(JSON.stringify({ outputFile, previewDir, sheets: sheets.map(([name]) => name) }));
