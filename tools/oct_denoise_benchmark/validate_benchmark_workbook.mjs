import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputFile = path.resolve(process.argv[2]);
const outputFile = path.resolve(process.argv[3] ?? `${inputFile}.validation.json`);
const input = await FileBlob.load(inputFile);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheets = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  options: { maxResults: 100 },
  maxChars: 12000,
});
const macro = await workbook.inspect({
  kind: "table",
  range: "三数据集宏平均!A1:U7",
  tableMaxRows: 7,
  tableMaxCols: 21,
  maxChars: 24000,
});
const acceptance = await workbook.inspect({
  kind: "table",
  range: "验收!A1:C11",
  tableMaxRows: 11,
  tableMaxCols: 3,
  maxChars: 12000,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 200 },
  summary: "post-import formula error scan",
});
await fs.writeFile(outputFile, JSON.stringify({ sheets, macro, acceptance, errors }, null, 2), "utf8");
console.log(JSON.stringify({ inputFile, outputFile }));
