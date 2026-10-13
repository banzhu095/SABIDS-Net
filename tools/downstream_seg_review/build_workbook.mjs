import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const root=process.argv[2];
const data=JSON.parse(await fs.readFile(path.join(root,'workbook_data.json'),'utf8'));
const wb=Workbook.create();
const names={display_name:'方法',method_id:'method_id',position_id:'位置',segmentation_seed:'分割seed',denoiser_seed:'降噪seed',
  layer_dice:'Layer Dice',vessel_dice:'Vessel Dice（全图）',vessel_roi_dice:'Vessel ROI Dice（GT层内）',
  precision:'Precision',recall:'Recall',iou:'IoU',boundary_dice:'Strict Boundary Dice',vessel_assd:'Vessel ASSD',
  vessel_outside_gt_layer_fraction:'层外预测血管占比',vessel_outside_gt_layer_fp:'层外血管FP率',
  upper_boundary_mae:'上边界MAE',lower_boundary_mae:'下边界MAE',thickness_mae:'厚度MAE',
  delta_full_dice:'全图Dice差值',delta_roi_dice:'ROI Dice差值',outside_penalty_reduction:'层外Dice惩罚减少',
  best_epoch:'best epoch',best_checkpoint_updates:'best checkpoint实际updates',total_training_updates:'总训练updates'};
const preview=path.join(root,'workbook_preview');await fs.mkdir(preview,{recursive:true});
for(const tab of data.tabs){
  const sheet=wb.worksheets.add(tab.name),cols=tab.columns;
  const headers=cols.map(c=>{
    const base=c.key.replace(/_(mean|std)$/,'');
    if(!names[base])return c.label;
    return names[base]+(c.key.endsWith('_std')?'（种子间SD）':'')+(/mae|assd/.test(base)?' (px)':'');
  });
  const values=[headers,...tab.rows.map(row=>cols.map(c=>row[c.key]??'不适用'))];
  sheet.getRangeByIndexes(3,1,values.length,cols.length).values=values;
  sheet.getRange('B2').values=[[tab.name]];sheet.getRange('B2').format.font={name:'Arial',size:14,bold:true};
  const used=sheet.getRangeByIndexes(3,1,values.length,cols.length);
  used.format.font={name:'Arial',size:10};used.format.rowHeight=22;
  used.format.columnWidth=18;used.format.verticalAlignment='center';
  sheet.getRangeByIndexes(3,1,1,cols.length).format={fill:'#34495E',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},
      wrapText:true,rowHeight:54,horizontalAlignment:'center'};
  sheet.getRangeByIndexes(4,1,tab.rows.length,1).format.columnWidth=38;
  for(let i=0;i<cols.length;i++){
    const c=cols[i],range=sheet.getRangeByIndexes(4,1+i,tab.rows.length,1);
    if(tab.rows.some(row=>typeof row[c.key]==='number')){range.setNumberFormat(c.format);range.format.horizontalAlignment='right';}
    if(/sha256|checkpoint_metadata_status|architecture/.test(c.key))range.format.columnWidth=72;
    if(/scope|position_id|segmentation_seed/.test(c.key))range.format.columnWidth=20;
  }
  sheet.showGridLines=false;sheet.freezePanes.freezeRows(4);sheet.freezePanes.freezeColumns(2);
  if(tab.name==='主结果'){
    const all=tab.rows.filter(r=>r.display_name!=='Clean reference');
    for(let i=0;i<cols.length;i++){
      const c=cols[i];if(!c.key.endsWith('_mean'))continue;
      const vals=all.map(r=>r[c.key]).filter(v=>typeof v==='number');if(!vals.length)continue;
      const lower=/mae|assd|outside/.test(c.key);const best=lower?Math.min(...vals):Math.max(...vals);
      tab.rows.forEach((r,j)=>{
        const cell=sheet.getRangeByIndexes(4+j,1+i,1,1);
        if(r.display_name==='Clean reference')cell.format.fill='#E8EDF4';
        else if(r[c.key]===best){cell.format.font.bold=true;cell.format.fill='#E6F0E7';}
      });
    }
  }
  const noteRow=values.length+6;
  sheet.getRangeByIndexes(noteRow,1,1,1).values=[['来源：'+tab.source+'；百分数均值/种子间SD；差值pp；空间误差px。绿色为非Clean最优，灰蓝为Clean参考。不适用不填0。'+(tab.name==='协议与资产索引'?'预测资产完整索引见prediction_export_manifest.csv。':'')]];
  sheet.getRangeByIndexes(noteRow,1,1,1).format.columnWidth=38;
  const blob=await wb.render({sheetName:tab.name,range:`B2:${String.fromCharCode(66+Math.min(7,cols.length)-1)}${Math.min(14,values.length+3)}`,scale:1.5,format:'png'});
  await fs.writeFile(path.join(preview,tab.name+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'主结果!B4:H14',include:'values,formulas',maxChars:3000,tableMaxRows:11,tableMaxCols:7})).ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!',options:{useRegex:true,maxResults:20},maxChars:1000});
console.log(errors.ndjson);
await (await SpreadsheetFile.exportXlsx(wb)).save(path.join(root,'downstream_seg_stage_summary.xlsx'));
await fs.writeFile(path.join(root,'workbook_verification.json'),JSON.stringify({backend:'artifact-tool',sheets:data.tabs.map(t=>t.name),
  exported:true,errorScan:errors.ndjson,typedNumeric:true,percentageFormat:true,sd:'between segmentation seeds',previewed:true},null,2));
