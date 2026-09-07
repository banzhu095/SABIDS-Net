import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const runDir = path.resolve(process.argv[2]);
const outputFile = path.resolve(process.argv[3] ?? path.join(runDir, "benchmark_summary.xlsx"));
const previewDir = path.join(runDir, "reports", "workbook_previews_protocol");

function parseCsv(text) {
  const rows = []; let row = []; let field = ""; let quoted = false;
  for (let i = 0; i < text.replace(/^\uFEFF/, "").length; i += 1) {
    const source = text.replace(/^\uFEFF/, ""); const ch = source[i];
    if (quoted) { if (ch === '"' && source[i + 1] === '"') { field += '"'; i += 1; } else if (ch === '"') quoted = false; else field += ch; }
    else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n") { row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = ""; }
    else field += ch;
  }
  if (field.length || row.length) { row.push(field.replace(/\r$/, "")); rows.push(row); }
  return rows.filter(r => r.some(v => v !== ""));
}

async function csv(relative) {
  try { return parseCsv(await fs.readFile(path.join(runDir, relative), "utf8")); }
  catch { return [["status", "detail"], ["missing", relative]]; }
}

function columnName(index) { let value = index + 1; let name = ""; while (value) { const part = (value - 1) % 26; name = String.fromCharCode(65 + part) + name; value = Math.floor((value - 1) / 26); } return name; }
function typed(rows) {
  if (!rows.length) return [["status"], ["empty"]];
  return rows.map((row, ri) => row.map((value, ci) => {
    if (ri === 0 || value === "") return value;
    if (/^-?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(value) && !/sha|path|id|frame/i.test(rows[0][ci] ?? "")) return Number(value);
    return value;
  }));
}

const overview = [
  ["项目", "值"],
  ["实验", "SABIDS-Net OCT 降噪比较"],
  ["运行标识", path.basename(runDir)],
  ["当前状态", "工程 smoke 已完成；正式校准、三种子训练和封存测试未完成"],
  ["数据协议", "PKU37 train/validation/test=25/6/6 positions；Duke17/28 全量 external test"],
  ["统计", "frame→position→dataset；bootstrap 10,000，seed=42"],
  ["重要限制", "本机无 CUDA；禁止把 smoke checkpoint 或 validation-only noisy baseline 当论文结果"],
];
const definitions = [
  ["指标", "方向", "说明"], ["PSNR / SSIM / MS-SSIM", "越高越好", "data_range=1.0"],
  ["RMSE / MAE / edge MAE / gradient MAE", "越低越好", "与 averaged/high-SNR reference 比较"],
  ["HF / Laplacian energy ratio", "接近1", "低值提示平滑，高值可能残留噪声"],
  ["abs(log energy ratio)", "越低越好", "对高于/低于 reference 对称惩罚"],
];
const specs = [
  ["说明", overview], ["指标定义", definitions], ["数据审计", await csv("audit/dataset_inventory.csv")],
  ["方法来源", await csv("audit/method_inventory.csv")], ["验收", await csv("audit/acceptance_checks.csv")],
  ["数据集结果", await csv("metrics/per_dataset_metrics.csv")], ["逐位置结果", await csv("metrics/per_position_metrics.csv")],
  ["逐图结果", await csv("metrics/per_image_metrics.csv")], ["配对差值", await csv("metrics/paired_method_differences.csv")],
  ["置信区间", await csv("metrics/bootstrap_confidence_intervals.csv")], ["参数搜索", await csv("metrics/parameter_search_results.csv")],
  ["参数选择", await csv("metrics/selected_parameters.csv")], ["Checkpoint", await csv("metrics/checkpoint_inventory.csv")],
  ["训练曲线", await csv("metrics/training_curves.csv")], ["运行时间", await csv("metrics/runtime_summary.csv")],
  ["模型复杂度", await csv("metrics/model_complexity.csv")], ["失败记录", await csv("failures.csv")],
];

const workbook = Workbook.create();
for (const [index, [name, rawRows]] of specs.entries()) {
  const rows = typed(rawRows); const sheet = workbook.worksheets.add(name); const width = Math.max(...rows.map(r => r.length));
  const normalized = rows.map(r => [...r, ...Array(width - r.length).fill("")]);
  sheet.getRangeByIndexes(0, 0, normalized.length, width).values = normalized;
  const lastCol = columnName(width - 1); const used = sheet.getRange(`A1:${lastCol}${normalized.length}`);
  sheet.showGridLines = false; if (normalized.length > 8) sheet.freezePanes.freezeRows(1);
  used.format.font = { name: "Arial", size: 10, color: "#1F2937" }; used.format.verticalAlignment = "top";
  const header = sheet.getRange(`A1:${lastCol}1`); header.format.fill = "#1F4E78"; header.format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }; header.format.rowHeight = 28; header.format.wrapText = true;
  if (normalized.length > 1 && name !== "说明" && name !== "指标定义") sheet.tables.add(`A1:${lastCol}${normalized.length}`, true, `ProtocolTable${index + 1}`);
  normalized[0].forEach((label, ci) => {
    const col = columnName(ci); const lower = String(label).toLowerCase(); let w = 15;
    if (/path|config|error|detail|limitation/.test(lower)) w = 38; else if (/method|dataset|position|status|metric|checkpoint/.test(lower)) w = 21; else if (/sha/.test(lower)) w = 28;
    sheet.getRange(`${col}:${col}`).format.columnWidth = w;
    if (/psnr|ssim|rmse|mae|ratio|epi|seconds|mean|std|ci95|gflops/.test(lower) && normalized.length > 1) sheet.getRange(`${col}2:${col}${normalized.length}`).format.numberFormat = "0.0000";
    if (/count|parameters|flops|bytes|seed|width|height|iterations/.test(lower) && normalized.length > 1) sheet.getRange(`${col}2:${col}${normalized.length}`).format.numberFormat = "#,##0";
  });
  if (name === "说明" || name === "指标定义") { sheet.getRange("A:A").format.columnWidth = 28; sheet.getRange("B:B").format.columnWidth = 78; used.format.wrapText = true; used.format.autofitRows(); }
  if (name === "指标定义") { sheet.getRange("B:B").format.columnWidth = 20; sheet.getRange("C:C").format.columnWidth = 58; }
  if (name === "模型复杂度") { sheet.getRange("B:B").format.columnWidth = 42; sheet.getRange("H:H").format.columnWidth = 32; }
  if (name === "训练曲线") { sheet.getRange("G:H").format.columnWidth = 26; }
  if (name === "方法来源") { sheet.getRange("A:A").format.columnWidth = 22; sheet.getRange("B:B").format.columnWidth = 42; sheet.getRange("C:E").format.columnWidth = 25; sheet.getRange("F:F").format.columnWidth = 62; used.format.wrapText = true; used.format.autofitRows(); }
  if (name === "验收") { sheet.getRange("A:A").format.columnWidth = 54; sheet.getRange("B:B").format.columnWidth = 18; sheet.getRange("C:C").format.columnWidth = 78; used.format.wrapText = true; used.format.autofitRows(); }
  if (name === "参数选择") { sheet.getRange("A:B").format.columnWidth = 24; sheet.getRange("C:C").format.columnWidth = 76; }
  if (name === "Checkpoint") { sheet.getRange("A:B").format.columnWidth = 20; sheet.getRange("C:C").format.columnWidth = 90; sheet.getRange("D:D").format.columnWidth = 70; sheet.getRange("F:F").format.columnWidth = 28; used.format.wrapText = true; used.format.autofitRows(); }
  if (name === "失败记录") { sheet.getRange("A:B").format.columnWidth = 28; sheet.getRange("C:C").format.columnWidth = 82; sheet.getRange("D:E").format.columnWidth = 18; used.format.wrapText = true; used.format.autofitRows(); }
}

const inspect = await workbook.inspect({ kind: "sheet,table", maxChars: 8000, tableMaxRows: 5, tableMaxCols: 8 });
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
await fs.mkdir(previewDir, { recursive: true });
for (const [name] of specs) { const blob = await workbook.render({ sheetName: name, autoCrop: "all", scale: 0.8, format: "png" }); await fs.writeFile(path.join(previewDir, `${name}.png`), new Uint8Array(await blob.arrayBuffer())); }
await fs.writeFile(path.join(runDir, "reports", "workbook_protocol_verification.json"), JSON.stringify({ inspect: inspect.ndjson, errors: errors.ndjson, sheets: specs.map(x => x[0]) }, null, 2));
const output = await SpreadsheetFile.exportXlsx(workbook); await output.save(outputFile);
console.log(JSON.stringify({ outputFile, sheets: specs.map(x => x[0]), previewDir }));
