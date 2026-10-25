import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

if(process.argv.length!==3)throw Error('Usage: node build_workbook.mjs <result-directory>');
const out=path.resolve(process.argv[2]);
const data=JSON.parse(await fs.readFile(path.join(out,'summary.json'),'utf8'));
const book=Workbook.create();
const main=book.worksheets.add('综合评分');
const raw=book.worksheets.add('原始指标');
const source=book.worksheets.add('逐折明细');
const keys=['local_order','local_mae','global_order','global_mae','top1_regret'];
const fields=['label','split_id',...keys.map(k=>'p_'+k),'score','local_mae','global_mae','local_bias','global_bias',
  'local_order_accuracy','global_order_accuracy','top1_regret','top1_actual','oracle_best_actual','top1_id','recall','low_n','test_n','train_std','source'];
const headers=['模型','外层分折','低值排序 /35','低值MAE /15','全局排序 /15','全局MAE /15','第一推荐 /20','总分',
  '低值MAE','全局MAE','低值偏差','全局偏差','低值排序正确率','全局排序正确率','第一推荐差距',
  '推荐真实值','测试最优值','第一推荐ID','原始召回率','优秀样本数','测试样本数','训练标准差s','来源'];
const col=n=>{let t='';for(let x=n+1;x>0;x=Math.floor((x-1)/26))t=String.fromCharCode(65+(x-1)%26)+t;return t;};
const ordered=data.summary.flatMap(model=>data.folds.filter(r=>r.group===model.group));
source.getRange(`A1:W111`).values=[headers,...ordered.map(r=>fields.map(k=>r[k]))];
function base(sheet,range) {
  sheet.showGridLines=false;sheet.getRange(range).format.font={name:'Arial',size:10};
  sheet.getRange(range).format.rowHeight=25;
}
function head(sheet,range) {
  sheet.getRange(range).format={fill:'#263E5A',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},
    horizontalAlignment:'center',verticalAlignment:'center',wrapText:true};sheet.getRange(range).format.rowHeight=40;
}
base(source,'A1:W111');head(source,'A1:W1');source.getRange('A1:A111').format.columnWidth=39;
source.getRange('B1:B111').format.columnWidth=13;source.getRange('C1:W111').format.columnWidth=18;
source.getRange('C2:V111').setNumberFormat('0.0000');
for(const c of ['M','N','S'])source.getRange(`${c}2:${c}111`).setNumberFormat('0.0%');
source.getRange('R2:R111').setNumberFormat('@');source.getRange('T2:U111').setNumberFormat('0');
source.freezePanes.freezeRows(1);source.freezePanes.freezeColumns(2);

base(main,'A1:L25');main.tabColor='#263E5A';
main.getRange('A1:A25').format.columnWidth=7;main.getRange('B1:B25').format.columnWidth=39;
main.getRange('C1:L25').format.columnWidth=16;
main.getRange('A2').values=[['11组现成模型：冻结分折下的v3.2复评']];
main.getRange('A2').format.font={name:'Arial',size:16,bold:true,color:'#263E5A'};
main.getRange('A3').values=[['截图前5组与原决赛圈6组；每组10个外层，不重新选参或训练，测试集不裁剪。']];
main.getRange('A3').format.font={name:'Arial',size:10,color:'#596676'};
main.getRange('A5:L5').values=[['排名','策略与模型','总分 /100','低值排序 /35','低值MAE /15','全局排序 /15','全局MAE /15','第一推荐 /20',
 '样本SD /n−1','总体SD /n','最低分','最高分']];head(main,'A5:L5');
const checks=[];
const formula=(sheet,cell,text,expected)=>{sheet.getRange(cell).formulas=[[text]];checks.push([sheet,cell,expected]);};
const avg=(key,i)=>{const c=col(fields.indexOf(key)),a=10*i+2;return `=AVERAGE('逐折明细'!${c}${a}:${c}${a+9})`;};
for(let i=0;i<11;i++) {
 const r=i+6,m=data.summary[i],a=10*i+2,b=a+9;
 main.getRange(`A${r}:B${r}`).values=[[i+1,m.label]];
 keys.forEach((k,j)=>formula(main,`${col(j+3)}${r}`,avg('p_'+k,i),m['p_'+k]));
 formula(main,`C${r}`,`=SUM(D${r}:H${r})`,m.score);
 formula(main,`I${r}`,`=STDEV.S('逐折明细'!H${a}:H${b})`,m.score_std*Math.sqrt(10/9));
 formula(main,`J${r}`,`=STDEV.P('逐折明细'!H${a}:H${b})`,m.score_std);
 formula(main,`K${r}`,`=MIN('逐折明细'!H${a}:H${b})`,m.score_min);
 formula(main,`L${r}`,`=MAX('逐折明细'!H${a}:H${b})`,m.score_max);
 if(i%2===1)main.getRange(`A${r}:L${r}`).format.fill='#F1F4F8';main.getRange(`A${r}:L${r}`).format.rowHeight=28;
}
main.getRange('C6:L16').setNumberFormat('0.00');main.getRange('C6:C16').format.font={name:'Arial',size:10,bold:true};
const notes=[
 '样本SD使用ddof=1，与截图一致；总体SD使用ddof=0，与原决赛圈报告一致。二者都不直接扣分。',
 '低值区统一为真实ramping≤132.80，0.5容差只用于排序；s来自完整、未筛选的外层训练标签。',
 'low60/80/90是训练策略，不是只评估部分测试数据。所有方案使用相同的完整冻结外层测试集。',
 '组员权重原先按RMSE选参，原决赛圈按v3.2选参。本表比较现成权重，不声称统一重新选参。',
 '组员50个分折文件均通过包内记录误差核对；原决赛圈60组留出预测重算与原记录完全一致。',
 `来源：分析记录${data.run_id}。源码未提交（本机未配置Git身份），源码与输入哈希已记录。`
];notes.forEach((text,i)=>{main.getRange(`A${19+i}`).values=[[text]];main.getRange(`A${19+i}`).format.font={name:'Arial',size:10,color:'#596676'};});

base(raw,'A1:K37');raw.getRange('A1:A37').format.columnWidth=39;raw.getRange('B1:K37').format.columnWidth=19;
raw.getRange('A2').values=[['原始测试指标：逐折计算后取10折平均']];raw.getRange('A2').format.font={name:'Arial',size:16,bold:true,color:'#263E5A'};
raw.getRange('A3').values=[['偏差=预测−真实；排序正确率为v3.2加权容差正确率；以下没有折算成分数。']];
raw.getRange('A3').format.font={name:'Arial',size:10,color:'#596676'};
raw.getRange('A5:I5').values=[['策略与模型','全局MAE ↓','低值MAE ↓','全局偏差','低值偏差','低值排序正确率 ↑','全局排序正确率 ↑','原始召回率 ↑','来源']];head(raw,'A5:I5');
const rawkeys=['global_mae','local_mae','global_bias','local_bias','local_order_accuracy','global_order_accuracy','recall'];
raw.getRange('A21:H21').values=[['策略与模型','第一推荐差距 ↓','差距标准差','最大差距','差距≤0.5折数','推荐真实值均值','测试最优值均值','截图总分']];head(raw,'A21:H21');
for(let i=0;i<11;i++) {
 const m=data.summary[i],r=i+6,t=i+22,a=10*i+2,b=a+9;raw.getRange(`A${r}`).values=[[m.label]];raw.getRange(`A${t}`).values=[[m.label]];
 rawkeys.forEach((k,j)=>{
  const cell=`${col(j+1)}${r}`;formula(raw,cell,avg(k,i),m[k]);
  raw.getRange(cell).setNumberFormat(k.includes('accuracy')||k==='recall'?'0.0%':k.includes('bias')?'+0.00;-0.00;0.00':'0.00');
 });
 raw.getRange(`I${r}`).values=[[m.source==='peer'?'组员模型':'原决赛圈']];
 formula(raw,`B${t}`,avg('top1_regret',i),m.top1_regret);
 formula(raw,`C${t}`,`=STDEV.P('逐折明细'!O${a}:O${b})`,m.top1_regret_std);
 formula(raw,`D${t}`,`=MAX('逐折明细'!O${a}:O${b})`,m.top1_regret_max);
 formula(raw,`E${t}`,`=COUNTIFS('逐折明细'!O${a}:O${b},"<=0.5")`,m.top1_within_half_count);
 formula(raw,`F${t}`,avg('top1_actual',i),m.top1_actual);formula(raw,`G${t}`,avg('oracle_best_actual',i),m.oracle_best_actual);
 raw.getRange(`H${t}`).values=[[m.screenshot_score]];
 raw.getRange(`B${t}:H${t}`).setNumberFormat('0.00');raw.getRange(`E${t}`).setNumberFormat('0');
 if(i%2===1){raw.getRange(`A${r}:I${r}`).format.fill='#F1F4F8';raw.getRange(`A${t}:H${t}`).format.fill='#F1F4F8';}
 raw.getRange(`A${r}:I${r}`).format.rowHeight=28;raw.getRange(`A${t}:H${t}`).format.rowHeight=28;
}
raw.getRange('A35').values=[['所有第一推荐均来自当前外层测试集，差距以该测试集真实最低值为基准，不调用模拟器。']];
raw.getRange('A36').values=[['截图总分在两位小数上全部复现；截图SD为样本SD，综合评分页同时列出两种口径。']];
raw.getRange('A37').values=[['各策略原先的选参目标不同；这些是现成模型在相同外层候选上的表现，不是新一轮训练结果。']];
raw.getRange('A35:K37').format.font={name:'Arial',size:10,color:'#596676'};
book.recalculate();
for(const [sheet,cell,expected] of checks) {
 const got=sheet.getRange(cell).values[0][0];if(typeof got!=='number'||Math.abs(got-expected)>1e-9)
  throw Error(`Mismatch ${sheet.name}/${cell}: ${got} vs ${expected}`);
}
console.log((await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!',options:{useRegex:true,maxResults:10},summary:'Eleven-model review error check'})).ndjson);
for(const [sheetName,range,file] of [['综合评分','A1:L25','workbook_score_preview.png'],['原始指标','A1:I37','workbook_raw_preview.png'],['逐折明细','A1:J12','workbook_fold_preview.png']]) {
 const image=await book.render({sheetName,range,scale:1.3,format:'png'});await fs.writeFile(path.join(out,file),new Uint8Array(await image.arrayBuffer()));
}
const xlsx=await SpreadsheetFile.exportXlsx(book);await xlsx.save(path.join(out,'eleven_model_review_v3_2.xlsx'));
await fs.writeFile(path.join(out,'workbook_validation.json'),JSON.stringify({status:'passed',formula_cells_checked:checks.length,rows:110},null,2));
console.log(`WORKBOOK COMPLETE: ${checks.length} formulas checked`);
