import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const runDir = path.resolve(process.argv[2]);
const outputFile = path.resolve(process.argv[3] ?? path.join(runDir, "benchmark_summary.xlsx"));
const previewDir = path.join(runDir, "reports", "workbook_previews");

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  const source = text.replace(/^\uFEFF/, "");
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (quoted) {
      if (char === '"' && source[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows.filter((entry) => entry.some((value) => value !== ""));
}

function toCsv(rows) {
  return rows
    .map((row) => row.map((value) => {
      const text = String(value ?? "");
      return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    }).join(","))
    .join("\r\n");
}

async function readCsv(relativePath) {
  return parseCsv(await fs.readFile(path.join(runDir, relativePath), "utf8"));
}

function filterRows(rows, column, expected) {
  const index = rows[0].indexOf(column);
  if (index < 0) throw new Error(`Missing column ${column}`);
  return [rows[0], ...rows.slice(1).filter((row) => row[index] === expected)];
}

function columnName(index) {
  let value = index + 1;
  let name = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    name = String.fromCharCode(65 + remainder) + name;
    value = Math.floor((value - 1) / 26);
  }
  return name;
}

const perDataset = await readCsv("metrics/per_dataset_metrics.csv");
const overall = await readCsv("metrics/overall_metrics.csv");
const sheets = [
  ["说明", [
    ["项目", "值"],
    ["工作簿用途", "SABIDS-Net 经典 OCT 降噪基线的可追溯汇总"],
    ["运行目录", runDir],
    ["数据范围", "仅 train/validation；sealed test 未用于本次推理与汇总"],
    ["主统计单位", "先 frame→position，再对数据集做等权宏平均"],
    ["参数选择", "仅 validation；按 position-macro PSNR，SSIM 用于并列判定"],
    ["主结果来源", "metrics/overall_metrics.csv 中 aggregation=dataset_macro"],
    ["置信区间", "位置级 bootstrap，10,000 次，seed=42"],
    ["生成时间", `UTC ${new Date().toISOString()}`],
  ]],
  ["方法清单", await readCsv("audit/method_inventory.csv")],
  ["环境版本", await readCsv("audit/environment_versions.csv")],
  ["数据清单", await readCsv("audit/dataset_inventory.csv")],
  ["PKU37结果", filterRows(perDataset, "dataset", "PKU37")],
  ["Duke17结果", filterRows(perDataset, "dataset", "Duke17")],
  ["Duke28结果", filterRows(perDataset, "dataset", "Duke28")],
  ["三数据集宏平均", filterRows(overall, "aggregation", "dataset_macro")],
  ["逐位置结果", await readCsv("metrics/per_position_metrics.csv")],
  ["参数", await readCsv("metrics/selected_parameters.csv")],
  ["参数搜索", await readCsv("metrics/parameter_search_results.csv")],
  ["运行时间", await readCsv("metrics/runtime_summary.csv")],
  ["失败记录", await readCsv("failures.csv")],
  ["配对差值", await readCsv("metrics/paired_method_differences.csv")],
  ["置信区间", await readCsv("metrics/bootstrap_confidence_intervals.csv")],
  ["验收", await readCsv("audit/acceptance_checks.csv")],
];

const workbook = Workbook.create();
for (const [sheetIndex, [sheetName, rows]] of sheets.entries()) {
  await workbook.fromCSV(toCsv(rows), { sheetName });
  const sheet = workbook.worksheets.getItem(sheetName);
  const used = sheet.getUsedRange();
  const lastColumn = columnName(rows[0].length - 1);
  const lastRow = rows.length;
  if (lastRow >= 2) sheet.tables.add(`A1:${lastColumn}${lastRow}`, true, `BenchmarkTable${sheetIndex + 1}`);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  used.format.font = { name: "Aptos", size: 10, color: "#243447" };
  used.format.verticalAlignment = "top";
  used.format.borders = { preset: "all", style: "thin", color: "#D8E1E8" };
  used.format.wrapText = false;
  used.format.rowHeight = 20;
  const header = sheet.getRange(`A1:${lastColumn}1`);
  header.format.fill = "#164E63";
  header.format.font = { name: "Aptos Display", size: 10, bold: true, color: "#FFFFFF" };
  header.format.rowHeight = 30;
  header.format.wrapText = true;
  for (let rowIndex = 3; rowIndex <= lastRow; rowIndex += 2) {
    sheet.getRange(`A${rowIndex}:${lastColumn}${rowIndex}`).format.fill = "#F3F8FA";
  }
  rows[0].forEach((rawHeader, index) => {
    const label = String(rawHeader).toLowerCase();
    const column = columnName(index);
    let width = 15;
    if (/path|reason|message|parameter|citation|license|notes|config|detail|provenance|algorithm/.test(label)) width = 42;
    else if (/method|dataset|position|sample|aggregation|reference|check|status/.test(label)) width = 20;
    else if (/sha256|hash/.test(label)) width = 30;
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
    if (/psnr|ssim|rmse|mae|epi|energy|ratio|gradient|laplacian|mean|median|seconds|ci95|difference/.test(label)) {
      sheet.getRange(`${column}2:${column}${Math.max(2, lastRow)}`).format.numberFormat = "0.0000";
    }
    if (/count|^n$|iterations|seed|height|width|bytes|images/.test(label)) {
      sheet.getRange(`${column}2:${column}${Math.max(2, lastRow)}`).format.numberFormat = "0";
    }
  });
  if (sheetName === "说明") {
    sheet.getRange("A:A").format.columnWidth = 24;
    sheet.getRange("B:B").format.columnWidth = 72;
    used.format.wrapText = true;
    used.format.autofitRows();
  }
  if (sheetName === "方法清单") {
    used.format.wrapText = true;
    sheet.getRange("B:B").format.columnWidth = 34;
    sheet.getRange("C:C").format.columnWidth = 30;
    sheet.getRange("D:D").format.columnWidth = 42;
    sheet.getRange("G:R").format.columnWidth = 30;
    used.format.autofitRows();
  }
  if (sheetName === "环境版本") {
    sheet.getRange("C:C").format.columnWidth = 38;
    used.format.wrapText = true;
    used.format.autofitRows();
  }
  if (sheetName === "参数") {
    used.format.wrapText = true;
    sheet.getRange("B:B").format.columnWidth = 24;
    sheet.getRange("C:C").format.columnWidth = 38;
    sheet.getRange("H:H").format.columnWidth = 42;
    sheet.getRange("I:I").format.columnWidth = 28;
    sheet.getRange("J:J").format.columnWidth = 58;
    used.format.autofitRows();
  }
  if (sheetName === "失败记录") {
    used.format.wrapText = true;
    sheet.getRange("G:G").format.columnWidth = 58;
    used.format.autofitRows();
  }
  if (sheetName === "配对差值") {
    sheet.getRange("B:C").format.columnWidth = 28;
  }
  if (sheetName === "验收") {
    used.format.wrapText = true;
    sheet.getRange("A:A").format.columnWidth = 36;
    sheet.getRange("C:C").format.columnWidth = 86;
    used.format.autofitRows();
  }
}

const inspect = await workbook.inspect({
  kind: "sheet,region,formula",
  maxChars: 16000,
  tableMaxRows: 6,
  tableMaxCols: 10,
  options: { maxResults: 200 },
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 200 },
  summary: "formula error scan",
});
await fs.writeFile(
  path.join(runDir, "reports", "workbook_verification.json"),
  JSON.stringify({ generated_at: new Date().toISOString(), sheets: sheets.map(([name]) => name), inspect, errors }, null, 2),
  "utf8",
);

await fs.mkdir(previewDir, { recursive: true });
for (const [sheetName] of sheets) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 0.75, format: "png" });
  await fs.writeFile(path.join(previewDir, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputFile);
console.log(JSON.stringify({ outputFile, previewDir, sheets: sheets.map(([name]) => name) }));
process.exit(0);
