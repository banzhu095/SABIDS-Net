import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const root=process.argv[2];
const tables=JSON.parse(await fs.readFile(path.join(root,'workbook_tables.json'),'utf8'));
const wb=Workbook.create();
const formal=(await fs.readFile(path.join(root,'workbook_mode.json'),'utf8').catch(()=>'{}'));
const filename=JSON.parse(formal).formal?'benchmark_summary.xlsx':'pilot_summary.xlsx';
await fs.mkdir(path.join(root,'workbook_preview'),{recursive:true});
for(const [name,rows] of Object.entries(tables)){
  const sheet=wb.worksheets.add(name); const columns=Object.keys(rows[0]);
  const values=[columns,...rows.map(r=>columns.map(c=>r[c]??null))];
  sheet.getRangeByIndexes(0,0,values.length,columns.length).values=values;
  const used=sheet.getUsedRange();used.format.font={name:'Arial',size:11};
  used.format.columnWidth=25;used.format.rowHeight=24;
  sheet.getRangeByIndexes(0,0,1,columns.length).format={fill:'#34495E',font:{bold:true,color:'#FFFFFF'},rowHeight:38,wrapText:true};
  if(name==='Status'){sheet.getRange('B1:B6').format.columnWidth=90;sheet.getRange('B1:B6').format.wrapText=true;sheet.getRange('A1:B6').format.rowHeight=48;}
  sheet.showGridLines=false;
  sheet.freezePanes.freezeRows(1);
  if(values.length>1){
    columns.forEach((c,i)=>{
      if(rows.some(r=>typeof r[c]==='number'))sheet.getRangeByIndexes(1,i,values.length-1,1).setNumberFormat(/seed|positions|updates|expected|available/.test(c)?'0':'0.0000');
    });
  }
  const preview=await wb.render({sheetName:name,range:`A1:${String.fromCharCode(64+Math.min(6,columns.length))}${Math.min(10,values.length)}`,scale:1,format:'png'});
  await fs.writeFile(path.join(root,'workbook_preview',name+'.png'),new Uint8Array(await preview.arrayBuffer()));
}
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'Status!A1:B6',include:'values',maxChars:2000,tableMaxRows:6,tableMaxCols:2})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:10},maxChars:1000})).ndjson);
await (await SpreadsheetFile.exportXlsx(wb)).save(path.join(root,filename));
await fs.writeFile(path.join(root,'workbook_export_verified.json'),JSON.stringify({filename,sheets:Object.keys(tables),exported:true}));
