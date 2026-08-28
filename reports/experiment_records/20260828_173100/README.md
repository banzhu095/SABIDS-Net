# SABIDS 实验记录归档

- 范围：records-only、validation-only；未运行训练、推理或test。
- 当前正式运行：E3b、E3-current、E3b-no-D2S、E1-current；历史E0–E3仅按找到的CSV单列。
- 当前四组使用相同manifest/effective split/标签raw+decoded/Stage 1初始化指纹；E3b vs E3-current、E3b vs no-D2S为声明的单因素对照，E1-current为多因素基线。
- E3b由metadata确认best epoch=52，history固定0.5 vessel Dice=0.742742。
- V0仅3个validation group、每组f01一帧，输入512×512、评价640×640；不是完整141帧验证。
- raw阈值校准最佳=0.425（512坐标）；V0使用0.35（640坐标），两者评价条件不一致。
- P3三帧共删除TP=101、FP=3749；收益限于固定帧，预测层外为0是算法构造。

## 复现

```bash
python tools/export_experiment_record.py --project-root .   --source-archive stage2_validity_fold0_20260828.tar.gz   --output reports/experiment_records/<timestamp>   --legacy-history E0=stage2_overfit_safe_fold0.csv   --legacy-history E1-old=stage2_segment_safe_fold0.csv   --legacy-history E2-old=stage2_segment_d2s_fold0.csv   --legacy-history E3-old=stage2_segment_roi_fold0.csv   --legacy-history E3b-old=stage2_segment_roi_outside_fold0.csv   --strict-val-only --records-only
```

`experiment_record.xlsx`由生成后的CSV构建。所有NA/unknown保留，不以0填充。
