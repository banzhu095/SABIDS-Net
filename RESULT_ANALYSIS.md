# 当前Joint结果分析与修复方案

## 0. 2026-08-27 Stage 2非有限损失补充诊断

新鲜Stage 2运行在epoch 12--19连续报告
`vessel_soft_dice=0.59735`，并在epoch 20训练中出现`loss=nan`。固定的
soft Dice提示模型几乎没有继续更新，但它本身不能证明输出就是整层；
必须联合查看Precision、Recall、预测血管面积和层-血管相似度。

代码审计发现层内基质负样本项原先使用
`-log(1-sigmoid(vessel_logit)).clamp(...)`。当正logit足够大时，sigmoid
在浮点数中舍入为1，clamp分支的梯度为0，导致整层假阳性进入饱和区后
最关键的纠正项失效。自定义Dice/Tversky归约在AMP下也没有显式提升到
FP32，增加了512x512训练的数值风险。

当前修复为：

- 基质负类BCE改为数学等价、梯度稳定的`softplus(vessel_logit)`；
- Dice、Tversky、stroma、area和containment使用FP32计算；
- Stage 2学习率降至`5e-5`并关闭AMP；
- 首个非有限loss立即打印样本和所有损失分量并停止；
- validation新增血管Precision/Recall、预测/真实面积比例及
  `pred_layer_vessel_dice`。

这只是代码层修复，尚未产生新的云端训练结果。旧Stage 2 checkpoint已
使用不同的损失定义，必须通过`--force`归档后从Stage 1重新训练，不能
resume。

## 1. 结论

当前降噪结果具有实际改善，但Joint分割发生了明显的血管假阳性扩张。`vessel_recall=0.8260`而`vessel_precision=0.3801`，并且预测血管面积比例为`0.7322`、真值仅`0.4152`。这说明模型不是“漏掉小血管”，而是把大量层内非血管组织也预测为血管。

本次命令日志显示`Reuse existing checkpoint`，因此该命令没有重新训练，只是用现有`best.pth`完成测试。旧流水线不会检查输入尺寸和checkpoint配置是否一致；v0.2已经修复。

## 2. Stage 2与Joint的变化

以下Stage 2数据来自前一轮同fold日志；如果两次运行使用了不同输入尺寸，则只能用于诊断趋势，不能作为论文中的严格配对比较。

| 指标 | Stage 2 | Joint | 变化 |
|---|---:|---:|---:|
| 层Dice | 0.8414 | 0.8045 | -0.0369 |
| 层Precision | 0.9639 | 0.7739 | -0.1900 |
| 层Recall | 0.7484 | 0.8394 | +0.0910 |
| 血管Dice | 0.6259 | 0.5205 | -0.1054 |
| 血管Precision | 0.5425 | 0.3801 | -0.1624 |
| 血管Recall | 0.7413 | 0.8260 | +0.0847 |
| 预测血管面积比例 | 0.5604 | 0.7322 | +0.1718 |
| 真值血管面积比例 | 0.4161 | 0.4152 | 基本相同 |

Joint同时出现Precision下降、Recall上升和预测面积大幅增加，符合“宽泛前景掩膜”退化，而不是单纯阈值波动。

## 3. 主要原因

1. 原血管Tversky参数为`FP=0.3, FN=0.7`，对漏检惩罚远大于误检，天然推动高Recall、低Precision。
2. 原containment仅抑制层外血管，无法阻止把整个脉络膜层预测成血管。
3. Joint使用全部公开位置均匀采样，只有13个PKU位置有血管真值，分割监督更新比例过低。
4. RMAC血管一致性过强时，会奖励跨重复帧稳定但过宽的血管掩膜。
5. Stage 2硬阈值验证Dice连续16轮完全相同，导致`best.pth`可能停留在第1轮，概率层面的学习没有被正确选择。
6. `256×256`可以做链路排错，但小血管暗腔的空间细节明显少于`512×512`。

## 4. v0.2代码修复

- repeat和clean教师采用stop-gradient紧凑前向，单步只保留main完整反向图；
- Joint关闭clean identity反向，避免第三套计算图；
- 血管Tversky改为`FP=0.6, FN=0.4`并加入BCE；
- 新增层内非血管基质负样本损失；
- 新增逐图像血管面积比例损失；
- containment优先使用真值层ROI，且阻断通过扩大层掩膜“作弊”的梯度；
- Joint中人工血管标注位置采样比例设为35%；
- RMAC总权重降到0.15，血管一致性权重低于层一致性；
- Joint学习率降到`5e-5`，ramp延长为30轮；
- 使用`vessel_soft_dice`选择最佳checkpoint；
- 新增验证集血管阈值校准工具；
- Stage 1评估不再输出未训练分割头的无意义指标；
- checkpoint复用前检查模型、输入尺寸、清单、采样和损失配置。

## 5. 推荐重训顺序

Stage 1降噪权重不需要重训。由于Stage 2和Joint的损失及选择准则已经变化，二者应重新训练：

```bash
cd /mnt/SABIDS-Net

python run_current_pipeline.py \
  --project-root /mnt/SABIDS-Net \
  --fold 0 \
  --stages segment joint \
  --device cuda \
  --batch-size 1 \
  --target-height 512 \
  --target-width 512 \
  --gradient-accumulation-steps 2 \
  --num-workers 4 \
  --skip-test \
  --force
```

`--force`会把旧结果移动到stage目录内的`archive_时间戳`，不会直接删除。如果512仍然OOM，把高度和宽度同时改为384；不要把256作为最终血管实验分辨率。

训练异常中断后，从上一个完整epoch继续：

```bash
python run_current_pipeline.py \
  --project-root /mnt/SABIDS-Net \
  --fold 0 \
  --stages joint \
  --device cuda \
  --batch-size 1 \
  --target-height 512 \
  --target-width 512 \
  --gradient-accumulation-steps 2 \
  --epochs-joint 120 \
  --num-workers 4 \
  --skip-test \
  --resume
```

旧版Joint checkpoint不能resume到新损失，应从Stage 2重新开始；v0.2训练产生的`last.pth`才可用上述命令续训。

## 6. 验证集阈值与测试

训练完成后只在validation split搜索阈值：

```bash
python tools/calibrate_vessel_threshold.py \
  --config runs/current/stage4_joint_fold0/resolved_config.yaml \
  --checkpoint runs/current/stage4_joint_fold0/best.pth \
  --split val \
  --output runs/current/stage4_joint_fold0/threshold_calibration
```

然后把输出JSON中的最佳阈值代入独立测试，例如：

```bash
python evaluate.py \
  --config runs/current/stage4_joint_fold0/resolved_config.yaml \
  --checkpoint runs/current/stage4_joint_fold0/best.pth \
  --split test \
  --vessel-threshold 0.625 \
  --output runs/current/stage4_joint_fold0/test_results_calibrated \
  --save-predictions
```

禁止在test split上选阈值。

## 7. 复核标准

重新训练后至少检查：

- `vessel_precision`不再明显低于Recall；
- `vessel_area_fraction_pred`接近真值，误差明显低于当前0.317；
- 新增的`pred_layer_vessel_dice`降低，说明两个预测不再近似相同；
- Joint血管Dice不应明显低于Stage 2；
- layer Dice、HD95和上下边界MAE不能因血管修复明显恶化；
- 降噪必须同时优于新增的`psnr_noisy/ssim_noisy/rmse_noisy`基线，而不只依赖定性观察。

数据准备后还应查看`Manifests/dataset_report.json`中的`area_statistics_by_label`。若某张人工标签的`vessel_fraction_of_layer`异常接近1，应先修正三分类标签值或标注导出方式，而不是继续调模型。

## 8. 若Joint仍退化，按顺序定位

以下配置都使用相同fold 0清单和Stage 2初始化，只改变一个因素：

```bash
python train.py --config configs/current/ablations/no_rmac_fold0.yaml
python train.py --config configs/current/ablations/no_ugbi_fold0.yaml
python train.py --config configs/current/ablations/denoise_to_seg_only_fold0.yaml
python train.py --config configs/current/ablations/seg_to_denoise_only_fold0.yaml
python train.py --config configs/current/ablations/no_uncertainty_fold0.yaml
python train.py --config configs/current/ablations/no_area_stroma_fold0.yaml
```

先比较`no_rmac`与完整模型：若血管Precision显著恢复，应继续降低RMAC血管一致性，而不是修改分割解码器。若`no_ugbi`恢复，则问题主要来自跨任务特征交互；再用两个单向配置判断是哪一个方向。`no_area_stroma`用于证明本次新增的层内约束是否真正抑制整层血管预测。
