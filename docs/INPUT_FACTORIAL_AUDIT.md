# I_NOISY / I_DENOISED 可归因输入实验

## 审计结论

本实验只改变分割网络看到的基础图像：`I_NOISY` 读取原始 noisy 的
float32 identity cache，`I_DENOISED` 读取冻结 Stage 1 D0 的 float32
输出。二者不使用 D→S、S→D、RMAC、pseudo、reconstruction、residual、
identity、阈值校准或后处理。实际 stage 名是 `input_segment`；共享 encoder
和 layer/vessel adapter、decoder、head 按完全相同规则训练，内部 denoise
adapter/decoder/head 和全部 UGBI 参数冻结且不进入 optimizer。

共同初始化不是已经适配 noisy 分割输入的 E3b checkpoint，而是同一个
fold-specific Stage 1 checkpoint 中、Stage 2 开始前的完整模型状态。工具把
该状态另存为 `preseg_initialization.pth`，两个分支分别从它完整训练。Stage 1
中尚未训练的 segmentation 参数也来自这个同一文件，不会分别重新随机初始化。
每个 seed 的初始化完整 state SHA、trainable 参数名 SHA、sampler/augmentation
计划 SHA 必须完全一致，否则 runner 在训练前停止。

完整 E3b Stage 2 预算经配置继承审计为 60 epoch、每 epoch 64 个 group-uniform
优化样本、batch size 1、512×512、AdamW/plateau、LR 5e-5。因此 I 实验使用
固定 60 轮，而不是 J 可行性实验的 20 轮。主实验只保留水平翻转；gamma 和
contrast 固定为 1，人工 speckle、blur 关闭，避免抹平 noisy/D0 的输入差异。
空间变换在 float cache 读取后执行，同 seed 的两组使用相同数据 RNG。

## D0 泄漏与缓存门禁

`prepare_input_factorial.py --mode audit` 从 checkpoint 的 resolved config 读取
Stage 1 实际 train groups、manifest SHA 和 effective split SHA，再与当前
fold-0 Stage 1 manifest 的 SHA 以及 segmentation val/test 的位置元数据比对。
它只打开/哈希 Stage 1 train/val noisy/clean 资产；test 只读 manifest 元数据，
不打开 test 图像、clean 或标签。以下任一情况都会标记 `blocked`，cache 与共同
初始化均不得继续：

- checkpoint 缺少 resolved config、train group 或 split/manifest 指纹；
- checkpoint manifest SHA 与当前 Stage 1 manifest 不一致；
- D0 train group 与当前 segmentation val/test group 重叠；
-当前 Stage 1 train/val noisy/clean 资产缺失。

兼容旧 checkpoint 时，仅接受同 run 目录中 `run_metadata.json`
作为旁证，且必须同时满足：`best_checkpoint_sha256` 与当前
`best.pth` 逐位一致，manifest/split/train-groups 指纹齐全，manifest SHA
与当前文件一致。`resolved_config.yaml`、路径相同或文件时间戳都不能
单独证明 checkpoint 的训练身份。

若阻断，应先按当前 fold 重新训练 D0：

```bash
python train.py --config configs/current/stage1_denoise_fold0.yaml
```

24 GB GPU 上的 fold-0 Stage 1 使用 512×512、物理 batch 1 和两步
梯度累积，保持等效 batch 2。Stage 1 专用 `forward_denoise_only`，
不计算尚未受监督的 layer/vessel/UGBI 分支；noisy reconstruction
和 clean identity 监督均保留。旧 OOM 运行不得 resume，因为它使用了
不同的物理 batch 和前向协议。
重建目标的 Charbonnier、MS-SSIM、wavelet、edge、residual 和
clean identity 均在 FP32 中计算，卷积前向仍使用 AMP。fold-0
的初始 loss scale 为 1024；偶发 AMP overflow 跳过该 optimizer
update 并下调 scale，连续超过 8 次才视为真实数值故障。

不要用其他历史 checkpoint 静默替换。`cache` 只处理 segmentation train/val，
D0 使用 `eval()` 和 `torch.no_grad()`，clean/GT 不进入 forward；模型同一输入
前向两次须在 1e-7 内一致。D0 的 512 resize/pad 输出先恢复到原图几何，再保存
float32 NPY。noisy 也保存为同格式 identity cache，逐元素 round-trip 必须精确
相等。已有 cache 仅在逐元素完全相同时复用，否则拒绝覆盖。

缓存审计输出包括文件/path/sample/group/D0 SHA、shape、dtype、范围、文件 SHA、
重复前向差异、identity 差异、实际字节数和重复帧 noisy/D0 MAE。训练读取 NPY
时只检查 finite 和 `[0,1]`，不再按图像最大值归一化，防止 double normalization。

## 固定评价和解释边界

主结果只读取固定第 60 轮 `last.pth`，完整 validation、恢复原图坐标、P0、
layer/vessel threshold=0.5、无 oracle gate/后处理/校准。frame 指标先在
`group_id` 内平均，再进行位置等权和 seed 汇总。test 始终封存。

小/中/大血管阈值使用 `derive_vessel_component_thresholds.py` 在训练位置的原始
标签网格上预注册三分位面积界限；不使用 512 resize 后的面积，也不输出未经
spacing 验证的微米值。相关性只作三个位置上的描述，不产生帧级显著性结论。

自动结论只允许以下方向：两任务同时提高；结果接近；layer 提高而 vessel
下降；两任务均下降；或混合结果。高 Precision 若伴随 Recall/小血管召回下降、
ROI 改善仅来自层外错误减少、暗管腔消失、厚度/下边界未改善，均会保留为权衡，
不能通过换阈值、checkpoint 或挑图制造正结论。

## 云端运行顺序

以下命令均在 `/mnt/SABIDS-Net` 执行。每个写入入口拒绝覆盖非空结果目录。

```bash
cd /mnt/SABIDS-Net

# 1. 交互历史/固定 checkpoint 审计（先运行下面的扰动诊断，再汇总）
for seed in 42 43 44; do
  for variant in j10 j11; do
    python tools/diagnose_interactions.py \
      --config runs/current/interaction_${variant}_fold0_seed${seed}/resolved_config.yaml \
      --checkpoint runs/current/interaction_${variant}_fold0_seed${seed}/last.pth \
      --output runs/current/interaction_${variant}_fold0_seed${seed}/dependence_diagnostics_v2 \
      --split val --device cuda \
      --groups pku_0006 pku_0012 pku_0040 \
      --component-size-thresholds "$SMALL_MAX" "$MEDIUM_MAX"
  done
done
python tools/analyze_interaction_suppression.py \
  --project-root . --seeds 42 43 44 \
  --output runs/interaction_suppression_diagnosis_v2
python tools/build_interaction_atlas.py --project-root . --seed 42 \
  --labelled-frames-per-group 1 \
  --output runs/interaction_factorial_report/atlas_all_labelled_v2

# 2. D0 泄漏审计；status 必须为 passed
python tools/prepare_input_factorial.py --project-root . \
  --stage1-checkpoint runs/current/stage1_denoise_fold0/best.pth --mode audit

# 3. 构建成对 float cache（train/val only）
python tools/prepare_input_factorial.py --project-root . \
  --stage1-checkpoint runs/current/stage1_denoise_fold0/best.pth \
  --mode cache --device cuda

# 4. 用训练位置原图坐标预注册 vessel component 面积分箱
python tools/derive_vessel_component_thresholds.py \
  --config configs/current/input_noisy_fold0.yaml \
  --output runs/current/input_factorial_common_fold0/component_thresholds.json
read SMALL_MAX MEDIUM_MAX < <(python -c \
  'import json; v=json.load(open("runs/current/input_factorial_common_fold0/component_thresholds.json"))["component_size_thresholds"]; print(*v)')
echo "original-grid component thresholds: $SMALL_MAX $MEDIUM_MAX"

# 5. 构建共同 pre-seg snapshot，再审计配对配置
python tools/prepare_input_factorial.py --project-root . \
  --stage1-checkpoint runs/current/stage1_denoise_fold0/best.pth --mode initialize
python tools/run_input_factorial.py --project-root . --mode audit \
  --seeds 42 43 44 --epochs 60 --device cuda

# 6. 单 seed 独立 smoke；写入 *_seed42_smoke，不占正式目录
python tools/run_input_factorial.py --project-root . --mode train \
  --seeds 42 --epochs 60 --device cuda --smoke-test

# 7. 三个配对 seed 的完整固定预算训练
python tools/run_input_factorial.py --project-root . --mode train \
  --seeds 42 43 44 --epochs 60 --device cuda

# 8. 固定第 60 轮完整 validation；保存统一预测图
python tools/run_input_factorial.py --project-root . --mode evaluate \
  --seeds 42 43 44 --epochs 60 --device cuda --save-predictions \
  --component-size-thresholds "$SMALL_MAX" "$MEDIUM_MAX"

# 9. 位置优先汇总和固定图册
python tools/summarize_input_factorial.py --project-root . \
  --seeds 42 43 44 --output runs/input_factorial_report_v1
python tools/build_input_factorial_atlas.py --project-root . --seed 42 \
  --output runs/input_factorial_report_v1/atlas_seed42
```

注意：第一段交互诊断也需要 `$SMALL_MAX/$MEDIUM_MAX`。若尚未生成 I cache，
可先完成步骤 2--4，再回来运行步骤 1；不能凭经验随意填写阈值。

## 主要输出

- `runs/interaction_suppression_diagnosis_v2/`：scale trajectory、feature/gate、
  clipping/group gradients、final scale、J10 位置归因、artifact audit、诊断 Markdown。
- `runs/current/input_factorial_common_fold0/`：D0 leakage/初始化审计、共同 snapshot、
  component thresholds 和 cache；cache 子目录含 manifest、质量指标、重复稳定性。
- `runs/input_factorial_registry/`：内容寻址的 I 实验注册表和每 seed 配对计划审计。
- `runs/current/input_{noisy,denoised}_fold0_seed*/`：resolved config、epoch-0 曲线、
  `last.pth` 和 fixed-final validation。
- `runs/input_factorial_report_v1/`：frame/position long CSV、逐位置/seed/总增益、
  输入质量关联、`SUMMARY.md`、失败清单和固定 HTML/PNG 图册。

本地仓库目前没有真实 Stage 1 best checkpoint，也没有云端 J histories/checkpoints，
因此只能完成代码级与合成 smoke，不能在本地声称 D0 无泄漏、交互压低原因或 I
实验数值收益。所有科学结论必须由上述云端产物生成。
