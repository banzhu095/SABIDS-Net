# 可归因的降噪—分割交互实验审计与运行协议

## 已确认的当前实现

- 当前公开联合阶段的真实名称是 `joint`，入口是 `run_current_pipeline.py --stages joint` 或 `train.py --config configs/current/stage4_joint_fold0.yaml`；仓库不存在 `stage3joint` 阶段。
- E3b-noD2S 的声明配置是 `configs/current/stage2_segment_roi_outside_no_d2s_fold0.yaml`。它使用 `512x512`、`Manifests/segmentation_folds/manifest_seg_fold0.csv`、Stage 1 fold-0 `best.pth` 初始化、`roi-bce-dice-outside-bce-no-d2s-v1` 标签/损失协议，并关闭 D→S 训练。真正云端 checkpoint 内的 resolved config、split 指纹、标签指纹和 Stage 1 初始化指纹仍必须由运行入口读取确认。
- 当前本机没有 `runs/current/stage2_segment_roi_outside_no_d2s_fold0/best.pth`。因此本机不能确认云端 checkpoint 的 epoch、实际 split、标签资产版本或 Stage 1 权重，也没有用其他历史权重替代。当前本机旧 segmentation manifest 是 train/val/test = 8/2/3 个位置、368/100/149 帧；这不能推定为云端实际 E3b-noD2S 协议。
- 参数边界：`stem + encoder_blocks + downsamples` 是共享编码器；`adapters/decoders.denoise + residual_head` 是去噪路径；`adapters/decoders.layer/vessel + layer/boundary/vessel_head` 是分割路径；`interactions.*` 内分别包含 S→D 和 D→S 映射及三个残差尺度。
- 历史 UGBI 在同一尺度从注入前特征并行计算两方向，没有显式环，但 S→D 使用逐尺度 `layer_head/vessel_head`。E3b 的 `auxiliary_weight=0`，这些内部概率头没有分割监督，不能作为可靠概率性能证据。
- 历史 `detach_cross` 同时阻断两方向来源；`detach_denoise_to_seg_source` 只阻断 D→S。第一轮新配置显式使用 `detach_d2s_source=true` 和 `detach_s2d_source=true`。
- `SABIDSLoss` 不是按非零 YAML 权重盲目累加，而是按 stage 选择 active terms。新的 `interaction` stage 实际只启用 reconstruction、residual、layer、vessel、vessel_outside、containment；stroma/area 权重固定为 0，identity/RMAC/pseudo 固定为 0，且没有 auxiliary segmentation loss。
- 评价先去除模型 padding，并可把连续概率恢复到原图网格后再以固定 0.5 阈值二值化；`valid_mask`、`label_valid_mask`、`vessel_valid_mask` 排除 padding/unknown。frame 指标先在 `group_id` 内平均，再对位置等权平均。公开输入训练坐标为 512，原图通常为 640；小血管分箱必须在恢复后的原图坐标重新定义。

## 第一轮实现

`J00/J10/J01/J11` 均执行同一无环骨架：

1. noisy 只编码一次，使用已训练的最终分割路径得到 S0 特征与最终概率头。
2. S→D 开启时，把 detach 的 S0 特征/概率逐尺度残差注入去噪路径。
3. D→S 开启时，把 detach 的最终去噪特征逐尺度残差注入第二次最终分割路径。
4. loss 只调用一次，只监督最终 denoised、layer 和 vessel 输出；clean/GT 从不作为模型 forward 输入。

四组都计算 S0、去噪和最终分割，关闭方向返回零注入，因此 decoder 次数、Dropout/随机数消耗和 loss 次数一致。共享编码器及其运行统计被冻结。交互尺度为零初始化；第一步允许只有 scale 得到梯度，scale 离开零后映射参数必须开始更新。

四组别名：J00=`d2s off/s2d off`，J10=`on/off`，J01=`off/on`，J11=`on/on`。每组独立从同一个 E3b-noD2S 文件加载，禁止串行微调。runner 对 checkpoint SHA、checkpoint 内协议、manifest、标签资产、split、模型初值和数据计划写指纹；缺失时直接失败且不搜索其他 run。B0 是该 checkpoint 在交互尺度清零后的零续训评价。

## 评价、诊断与预留实验

主分析固定 `last.pth`（第 20 轮）和 P0=0.5，同一 checkpoint 同时评价三任务。frame 先按解剖位置归约，再计算 seed 均值/SD。误差类符号翻转，使正值始终表示改善；Dice/Precision/Recall 用百分点，PSNR/SNR 用 dB，边界/厚度误差用 px。不把重复帧或三个 seed 当独立病例，不伪造微米值。

评价包含全图/GT 层 ROI 的 PSNR、SSIM、RMSE，参考边缘 MAE、已有 EPI，vessel/stroma CNR 相对 clean 绝对误差，以及组内重复帧 denoised MAE、layer/vessel Dice。定性 crop 只由 B0 的 GT/中心规则决定，所有模型共用。

`tools/diagnose_interactions.py` 生成同 checkpoint 单方向关闭、强度扫描、空间错位和跨位置引导，明确标记为“依赖/扰动诊断”。跨位置使用两个确定性不同位置的循环置换，不使用 batch-size-1 shuffle。

`configs/interaction_capacity_control.yaml` 预留接收分支自身特征容量控制。`configs/future/interaction_g*.yaml` 预留两方向 detach/open 的 G0/GD/GS/GDS，不在第一轮 runner 内。I-noisy/I-denoised 仍要求先审计冻结 D0 的训练数据泄漏并实现离线 D0 输入 manifest，当前未伪装成可运行配置。

## 运行顺序（Linux/矩池云）

```bash
cd /mnt/SABIDS-Net

python tools/run_interaction_factorial.py --project-root . --mode audit --seeds 42
python tools/run_interaction_factorial.py --project-root . --mode b0 --seeds 42 --device cuda --save-predictions

python tools/run_interaction_factorial.py --project-root . --mode train --seeds 42 --epochs 20 --device cuda
python tools/run_interaction_factorial.py --project-root . --mode evaluate --seeds 42 --epochs 20 --device cuda --save-predictions

# seed 42 已按同一固定预算完成，只补 43/44，避免覆盖 seed 42
python tools/run_interaction_factorial.py --project-root . --mode train --seeds 43 44 --epochs 20 --device cuda
python tools/run_interaction_factorial.py --project-root . --mode evaluate --seeds 43 44 --epochs 20 --device cuda --save-predictions

python tools/summarize_interaction_factorial.py --project-root . --seeds 42 43 44
python tools/build_interaction_atlas.py --project-root . --seed 42

python tools/diagnose_interactions.py \
  --config runs/current/interaction_j11_fold0_seed42/resolved_config.yaml \
  --checkpoint runs/current/interaction_j11_fold0_seed42/last.pth \
  --output runs/current/interaction_j11_fold0_seed42/dependence_diagnostics \
  --split val --device cuda
```

短链路验证仍要求真实 E3b-noD2S checkpoint 和 train/val manifest：

```bash
python tools/run_interaction_factorial.py --project-root . --mode train --seeds 42 --smoke-test --device cpu
```

## 成功与失败判据

工程成功要求：四组初始化 checkpoint SHA、完整模型 state SHA、共同参数 SHA、交互参数 SHA、同 seed 数据计划 SHA 一致；encoder 参数不更新；所有输出/loss/引导有限；启用方向的 scale 和随后映射参数更新；20 轮完整结束；四组相同 validation 位置齐全。

科学结果不预设为正。分别报告 J10-J00、J11-J01、J01-J00、J11-J10、J11-J00 和交互项。高 Recall 伴随低 Precision、vessel area 过大、层外 FP 或降噪恶化均视为失败/权衡，不能靠校准阈值、后处理、oracle ROI、best-epoch 拼接或选择性样本掩盖。
