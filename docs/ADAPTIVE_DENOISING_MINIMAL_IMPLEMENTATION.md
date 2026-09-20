# 降噪剂量—反应：第一阶段最小实现

本轮只验证工程闭环，不证明“适度降噪有利、过度降噪有害”。Oracle 使用
clean 参考，是非部署的参照实验。`a125` 仅称外推/残差放大，不称过度平滑。
不实施双视图、D2、控制器、Joint factorial、PCGrad、alpha 选择、完整图册或
正式统计检验；不读取 test 图像/标签，不修改现有 split 或历史 runs。

## 实际入口与约束

- `audit_adaptive_denoising_baseline.py`：显式正式 checkpoint + 唯一 active lock；
  元数据核验完成后才打开 train/val 资产。候选权重清单只展示，绝不自动选择。
- `prepare_dose_response_inputs.py`：缓存、逐臂 manifest、展开配置、共同注册表；
  默认不训练。只有明确的 `--cpu-smoke` 执行合成数据 1 轮检查。
- `train.py --config`：原 Trainer、原 `input_segment`；新功能仅由
  `dose_response.enabled` 启用。模板不是可直接训练的配置，必须先生成注册配置。
- `evaluate.py`：完整 validation、固定 P0 0.5、只评价 layer/vessel；显式
  `--no-restore-original-geometry`。新剂量配置禁止默认 test、EMA、按组挑一帧及阈值变更。

支持 alpha = 0/.25/.50/.75/1/1.25，编码 a000/a025/a050/a075/a100/a125。
Oracle = clip(noisy + alpha*(clean-noisy))；D1 = clip(noisy-alpha*(noisy-D1(noisy)))。
所有运算 float32，端点单独处理以免相减造成恒等关系舍入误差。
D1 只调用 checkpoint 的 `forward_denoise_only(noisy)`；clean/GT 不进入 D1 前向。

## Formal preflight：失败即阻塞

必须显式选择 `best_validation_psnr` 或 `fixed_final`，不接受 `auto`：

1. checkpoint 存在、非 smoke/pilot，embedded config 与旁边的
   `resolved_config.yaml` 的 model/loss/data/train/seed 一致。
2. stage=denoise、restoration_mode=structure_d1、配置预算至少 60 轮；
   fixed normalization、target size 与 active lock 一致。
3. active lock 唯一；5 项协议身份一致：protocol ID、data plan、label inventory、
   dataset inventory、split contract。缺失、不匹配、不可重构均阻塞。
4. 显式 `--split-contract` 文件 SHA 与 lock 一致，held-out IDs 一致。
   不猜测 contract 路径，不在 13/3 和 8/2/3 两套历史协议间选择。
5. checkpoint manifest 的历史 SHA、effective split SHA 必须能追溯；
   group/patient 不跨 split，D1 train 不包含 segmentation val/sealed test。
6. 当前实现核验现有 v3 表布局：`train_joint.csv` + `test_sealed.csv`
   的纯元数据按 image_path 重构 data plan；不生成新 split，不打开 sealed 资产。
   无法按这一布局证明的其他协议当前记 BLOCKED，不能替换成 v3。
7. `train_segment.csv` 必须是锁定 data plan 的同一 train/val 行，所有帧均有
   clean/layer/vessel；不为某条曲线悄悄筛除缺失帧。当前标签 hash 必须匹配锁定 inventory。
8. selection history 使用独立的逐行 CSV 审计，不通过 pandas 强制矩形化、不使用
   `on_bad_lines="skip"`，也不修改历史文件。两种预先冻结的选择规则含义不同：
   - `fixed_final` 只信任 `epoch`：行数和整数 epoch 必须严格等于完整的
     `1...configured_epochs`，checkpoint 必须是达到最终轮次的 `last.pth`。
     它**不读取、不要求、也不使用 `val_psnr` 选模**。即使旧 history 从 287 列
     漂移到 295 列，只要完整 epoch 证据成立，也可继续；报告必须记录行宽分布、
     `history_schema_drift_detected=true` 和 `val_psnr_trusted=false`。
   - `best_validation_psnr` 要求 schema 一致的矩形 history、完整 epoch 和逐行有限的
     `val_psnr`。`best.pth` 必须是 val PSNR 首次最大值的 epoch，monitor=psnr，且
     `run_metadata.json.best_checkpoint_sha256` 匹配。行宽漂移而文件自身没有对应新表头时
     无法恢复指标含义，必须 fail closed，不能把错位后的约 0.13 当成 PSNR。

   两种规则都拒绝缺失/重复/非整数或非有限 epoch 以及未完成预算。只有计划 60 轮、
   实际跑了 2 轮的 checkpoint 不能因为名字不含 smoke/pilot 就变为正式模型。
   其他早停协议需单独审计适配，当前不替代。
9. 必须有训练时留下的 noisy/clean 像素指纹。协议中的 dataset inventory SHA
   只是表指纹，不证明历史像素没变。缺失历史像素证据也阻塞，不追认、不自动重训。

历史像素证据优先取确实由旧训练流程嵌入 checkpoint 的
`runtime.train_val_noisy_clean_asset_sha256`。不能在训练后补写该字段。另一种输入是明确的
`--training-asset-inventory`：**必须来自训练开始时保留的不可变记录，并与训练结束的
checkpoint 绑定**。只有单个 `recorded_at_training=true` 文件不构成证据链。

新 D1 复现实验由 `training_asset_evidence.enabled: true` 显式启用；旧配置没有此键，
行为不变。Trainer 在 DataLoader 完成 split/dataset/group/Subset 实际筛选后、优化器创建及
首个 step 之前，以独占创建方式写 `training_asset_inventory_initial.json`。它只散列
train/val 的 `image_path`、`clean_path`，不读取 test。固定 60 轮正常完成并保存
`last.pth` 后，才独占创建 `training_asset_inventory_last.json`。绑定阶段要求初始文件存在、
内容与当前像素及 manifest 一致、checkpoint epoch 等于完整预算；resume 或已有
checkpoint/history 的目录不能启用该功能，因而不能给旧 checkpoint 事后补证。

正式绑定文件的核心 schema 为：

```json
{
  "recorded_at_training": true,
  "recorded_before_optimizer_step": true,
  "checkpoint_sha256": "所选checkpoint的真实SHA",
  "manifest_sha256": "该run训练manifest的真实SHA",
  "records_sha256": "records的stable_sha",
  "train_val_noisy_clean_asset_sha256": "records的stable_sha",
  "source_initial_evidence_file": "training_asset_inventory_initial.json",
  "source_initial_evidence_sha256": "初始证据文件SHA",
  "completed_epochs": 60,
  "selection_rule": "fixed_final",
  "records": [
    {"sample_id": "...", "split": "train", "column": "image_path", "sha256": "..."},
    {"sample_id": "...", "split": "train", "column": "clean_path", "sha256": "..."}
  ]
}
```

records 覆盖 D1 **实际筛选**的全部 train/val noisy/clean，按 `(sample_id,column)`
排序；stable_sha 是 UTF-8、sort_keys=True、separators=(",",":"), ensure_ascii=False
的标准 JSON SHA256。不能现在从当前像素生成这个文件并声称是历史证据。
历史证据只有另一种字段/格式时，先审计真实性再做显式适配，不降低门禁。

`configs/adaptive_denoising/d1_repro_pku37_v3_seed42.yaml` 继承旧 D1 模板，并在训练前
读取旧正式 run 的 `resolved_config.yaml` 做逐项语义比较。忽略运行时派生字段后，必须且
只能出现 `train.output_dir` 与整个显式 `training_asset_evidence` 配置两项差异；报告保存为
`d1_reproduction_semantic_audit.json`。其余模型、loss、manifest、30/3 split、512×512、
fixed normalization、AdamW/cosine、batch/accumulation、增强、seed 和 60 轮预算有任何差异
都会在开始训练前阻塞。fixed-final 主分析不依赖 val PSNR 选模。

旧正式 run 可能由启动器写入 active-lock 的 data-plan/label-inventory SHA，也可能因中断
续跑而在 resolved config 留下 `train.resume`。新复现会在语义审计前从显式 active lock
绑定这些协议 SHA，并在报告的 `protocol_lock_bindings` 中保留模板旧值与绑定值；随后仍
要求绑定后的身份与旧正式 resolved config 一致。`train.resume` 单列为
`operational_differences`：新 run 必须为 null，绝不能加载旧 run 的模型、optimizer 或
scheduler 状态；该差异不属于模型、数据、损失、优化器超参数或增强语义变化。

pth 使用 trusted-local `torch.load(weights_only=False)`，不要对来源不明的文件使用。
本轮没有修改历史 D1 Trainer 默认行为或给旧 checkpoint 回填指纹。

## 同一几何与配对训练

noisy/clean/layer/vessel/validity 原始 shape 必须相同；不隐式把 512 GT 升到 640。
共同保持长宽比 resize+中心 pad：图像缩小 AREA、放大 LINEAR；GT/validity NEAREST。
原始 noisy 尺寸来自原始资产而不是剂量缓存。metadata 记录原始 clean/GT 尺寸、
resize、pad、crop、模型输入/输出及实际评价形状。

D1 的 clipped 输出取共同有效内容、原位重新零填充 padding。alpha=1 在有效区域与
当前 clipped D1 完全相同，padding 本来不是图像内容。方形 640→512 没有 padding，
整张 alpha=1 就是 D1 输出。所有臂使用同一空间有效 mask、同一零填充规则。

训练/evaluation 仅在明确标记的 **model grid，单位 px**。
评价从共同空间 canvas 裁掉 pad，**不**恢复原图；原分辨率边界指标 NOT IMPLEMENTED。
annotation-invalid 像素不进入 BCE/Dice/ROI/outside/containment。含 unknown 的列不参加
boundary supervision；有 annotation-invalid 的帧不报告有歧义的边界/表面指标，记 NA，
而不是制造 unknown 区域的假边缘。二值 uint8 0/1 和 0/255 正确读取；二值 255 是
前景，unknown 必须另给 validity mask。多分类标签不允许静默当二值标签读取。

不同曲线/alpha 每组从头独立训练共享编码器+layer/vessel 解码器；相同 seed 对应
同一随机初始化，**不**顺序微调，不使用不同的旧 Stage2 初始化。
内部 denoise 分支、全部 interaction 参数冻结。旧完整 forward 仍会计算一些未使用的
恢复/内部头输出；没有额外注入/梯度/监督，不将这些恢复输出冒称剂量降噪结果。
无需修改 `sabids_net.py`。
新剂量损失的 zero_source=final_segmentation，零损失不连接内部恢复图；未使用的
恢复/辅助输出即使被测试注入 NaN，也不能污染最终分割目标。历史零损失默认保持不变。

显式关闭两个方向的 alias 和 legacy switches，causal_interaction_experiment=false，
auxiliary_weight=0，rec/residual/RMAC/identity/pseudo/stroma/area 权重=0。
共同 E3b 最终监督：layer=1、ROI vessel BCE+Dice=1、outside negative BCE=.5、
containment=.1；沿用最终 layer boundary 权重 .2，不加新监督。

数据 RNG 独立于模型 RNG；GroupUniformSampler 使用 seed+epoch 的独立 generator。
增强仅为由 SHA256(seed,epoch,sample_id) 确定的水平翻转，保存实际布尔计划 SHA。
严格最小模式 num_workers=0，禁止随机 strong augmentation；Dropout=0。
初始化审计保存完整 state、可训练参数、cohort、sampler、actual augmentation 的 SHA，
`data_plan.json` 保存逐 epoch sampler indices。各臂 optimizer/scheduler/预算一致。
默认正式 batch=1, accumulation=2, FP32, AdamW 5e-5, cosine，固定最终轮次为主结果；
best 按 val vessel_soft_dice 首次最大值为次要分析，所有任务用同一 checkpoint。
patience=epochs+1，不会提前终止剂量臂。

## 输出安全与复核文件

缓存：`cache/adaptive_denoising/<protocol_id>/dose_v1/`；训练：
`runs/adaptive_denoising/<protocol_id>/dose_v1/<budget>_s<seeds>_<tag>/<curve>_<alpha>_seed<seed>/`。
preparations 内有共同注册表、各臂 train/val manifests 和展开 YAML。
缓存 key 含 protocol/sample/group/curve/alpha、noisy/clean/label/checkpoint/config SHA、
geometry SHA、源码内容指纹（包括未提交更改），并记录生成时间/commit/clip 比例。
相同身份及数组只读复用；不同身份/content、缺失 sidecar、旧 registry/config 都拒绝覆盖。
重用同一个 preparation tag 需要参数/数据/代码/设备完全一致；改协议或代码必须新 tag。
训练前复验 registry/config/cache/formal gate，已有非空 run 拒绝启动；本阶段未实现 resume。
所有新 JSON allow_nan=False；不适用/非有限值为 null，不写 NaN/Infinity。

关键复核文件：`initialization_audit.json`, `parameter_audit.json`, `data_plan.json`,
`resolved_config.yaml`, `history.csv`, `epoch0.pth`, `last.pth`, `best.pth`,
`dose_training_metadata.json`, `diagnostics/val_groups_epochNNN.csv`。
完整离线 val 输出 `frame_metrics.csv`, `group_metrics.csv`, `summary.json`,
`metric_definitions.json`。按帧兼容指标与解剖位置等权主汇总复用原 evaluator；
重复帧不当独立病例，不在这阶段做 alpha 选择或正式统计推断。
剂量评价仅记录 layer/vessel 重复帧稳定性及 `repeat_dose_input_mae`；不会把内部冻结
恢复分支的随机输出写成实际降噪指标。共同 annotation-valid 区域用于跨任务/重复比较。

## 矩池云后续命令（本轮未执行）

先同步包含本功能的 `feature/adaptive-denoising-dose-v1` 分支。
下面所有变量必须由用户填写明确的实际路径，不是自动发现。建议 Python 3.10+，
使用已验证的项目环境；CUDA 环境本地未验证。

### 0. 复现带训练时资产证据的新 D1

历史 D1 缺少训练时 noisy/clean pixel inventory，禁止补写。先确认旧正式
`resolved_config.yaml`、active protocol lock 和当前 manifest 均存在；新配置会在创建模型和
optimizer 前审计新旧配置，并要求实际 DataLoader 的 train/validation 位置严格等于 lock。

```bash
cd /mnt/SABIDS-Net
set -euo pipefail
test -f runs/current/d1_structure_pku37_v3_fold0_seed42/resolved_config.yaml
test -f Manifests/pku37_binary_v3/active_protocol_lock.json
test -f Manifests/pku37_binary_v3/train_denoise.csv
test ! -e runs/adaptive_denoising/pku37_binary_v3/d1_repro_fold0_seed42
python train.py \
  --config configs/adaptive_denoising/d1_repro_pku37_v3_seed42.yaml
```

成功标准包括完整 60 轮、`last.pth` epoch=59，以及同一新 run 下同时存在：

```text
d1_reproduction_semantic_audit.json
training_asset_inventory_initial.json
training_asset_inventory_last.json
```

其中语义审计只能列出 `train.output_dir` 和 `training_asset_evidence` 两项差异；绑定文件
的 checkpoint SHA 必须等于 `last.pth`。缺少初始证据、提前终止、manifest/像素变化或
旧目录非空均为失败，不应通过 resume 或手工编辑 JSON 继续。

### 1. Formal audit

```bash
cd /mnt/SABIDS-Net
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export D1_CKPT='runs/adaptive_denoising/pku37_binary_v3/d1_repro_fold0_seed42/last.pth'
export SABIDS_PROTOCOL_LOCK='/替换为明确协议目录/active_protocol_lock.json'
export SABIDS_SPLIT_CONTRACT='/替换为锁定的split_contract.yaml'
export D1_SELECTION_RULE='fixed_final'
EVIDENCE_ARGS=(--training-asset-inventory \
  'runs/adaptive_denoising/pku37_binary_v3/d1_repro_fold0_seed42/training_asset_inventory_last.json')
PREFLIGHT_OUT="reports/adaptive_denoising/formal_$(date +%Y%m%d_%H%M%S)"
python -B tools/audit_adaptive_denoising_baseline.py \
  --project-root . --mode formal \
  --denoiser-checkpoint "$D1_CKPT" \
  --protocol-lock "$SABIDS_PROTOCOL_LOCK" \
  --split-contract "$SABIDS_SPLIT_CONTRACT" \
  --selection-rule "$D1_SELECTION_RULE" \
  "${EVIDENCE_ARGS[@]}" --output "$PREFLIGHT_OUT"
DOSE_PROTOCOL_ID=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["protocol"]["protocol_id"])' "$PREFLIGHT_OUT/preflight_report.json")
```

返回 2/status=blocked 就停下。修复证据/路径问题，不绕过门禁、不改 cohort。

### 2. 生成完整 train/validation 剂量缓存与 seed42 pilot 配置（不训练）

```bash
python -B tools/prepare_dose_response_inputs.py \
  --project-root . --mode formal --budget pilot --tag pilot_v1 \
  --denoiser-checkpoint "$D1_CKPT" --protocol-lock "$SABIDS_PROTOCOL_LOCK" \
  --split-contract "$SABIDS_SPLIT_CONTRACT" --selection-rule "$D1_SELECTION_RULE" \
  "${EVIDENCE_ARGS[@]}" \
  --curves oracle d1 --alphas 0 .25 .5 .75 1 1.25 --seeds 42 --device cuda
```

这里生成 12 臂配置，不训练、不选择 alpha。原始资产几何不匹配时停止，不临时缩放旧 GT。

### 3. 单 alpha 短 CUDA overfit/链路检查（需要另行执行）

```bash
python -B tools/prepare_dose_response_inputs.py \
  --project-root . --mode formal --budget overfit --tag cuda_check_v1 \
  --denoiser-checkpoint "$D1_CKPT" --protocol-lock "$SABIDS_PROTOCOL_LOCK" \
  --split-contract "$SABIDS_SPLIT_CONTRACT" --selection-rule "$D1_SELECTION_RULE" \
  "${EVIDENCE_ARGS[@]}" --curves d1 --alphas .5 --seeds 42 --device cuda
CUDA_CFG="cache/adaptive_denoising/$DOSE_PROTOCOL_ID/dose_v1/preparations/overfit_s42_cuda_check_v1/config_d1_a050_seed42.yaml"
python -B train.py --config "$CUDA_CFG"
CUDA_RUN="runs/adaptive_denoising/$DOSE_PROTOCOL_ID/dose_v1/overfit_s42_cuda_check_v1/d1_a050_seed42"
python - "$CUDA_RUN" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
r = Path(sys.argv[1]); m = json.loads((r/'dose_training_metadata.json').read_text())
h = pd.read_csv(r/'history.csv')
assert m['completed_epochs'] == 2
assert m['completed_optimizer_steps'] == m['expected_optimizer_steps'] == 2
assert m['changed_trainable_parameter_names'] and not m['changed_frozen_parameter_names']
assert np.isfinite(h.train_total).all() and np.isfinite(h.val_vessel_soft_dice).all()
assert (r/'last.pth').is_file() and (r/'parameter_audit.json').is_file()
print('CUDA chain check: PASS; NOT FOR SCIENTIFIC EVALUATION')
PY
```

这个最小 overfit budget 只有 2 train/2 val、2 轮、2 次 optimizer update，足以检查
显存、前后向/有限性与产物，不是充分收敛或能拟合小样本的证明。必须检查参数确实更新、
关闭路径未更新；若需真正拟合至高 Dice，另行授权增加调试预算，不修改正式臂预算。
OOM/NaN/未更新/元数据缺失均是 FAIL，不因改阈值或 postprocessing 变为 PASS。

### 4. Seed42 pilot（只有 formal/geometry/pairing/CUDA 门禁均 PASS 才执行）

```bash
for CURVE in oracle d1; do
  for ALPHA in a000 a025 a050 a075 a100 a125; do
    CFG="cache/adaptive_denoising/$DOSE_PROTOCOL_ID/dose_v1/preparations/pilot_s42_pilot_v1/config_${CURVE}_${ALPHA}_seed42.yaml"
    python -B train.py --config "$CFG"
  done
done
```

每臂固定 20 轮，从头独立初始化。比较所有臂 initial model、cohort、sampler 和 actual
augmentation SHA；必须一致。最后一轮为主，best 只作预先声明的次要分析。

完整 validation 离线评价示例（不是 test、不恢复原图）：

```bash
for CURVE in oracle d1; do
  for ALPHA in a000 a025 a050 a075 a100 a125; do
    RUN="runs/adaptive_denoising/$DOSE_PROTOCOL_ID/dose_v1/pilot_s42_pilot_v1/${CURVE}_${ALPHA}_seed42"
    python -B evaluate.py --config "$RUN/resolved_config.yaml" \
      --checkpoint "$RUN/last.pth" --split val --tasks layer vessel \
      --postprocess-modes p0 --no-restore-original-geometry \
      --output "$RUN/validation_final"
  done
done
```

不能重复使用非空输出目录。人工复核/后续分析前，不自动选择 alpha；无收益、任务受损照实记录。

### 5. 后续多 seed（条件式命令，本轮未授权执行）

```bash
python -B tools/prepare_dose_response_inputs.py \
  --project-root . --mode formal --budget full --tag full_v1 \
  --denoiser-checkpoint "$D1_CKPT" --protocol-lock "$SABIDS_PROTOCOL_LOCK" \
  --split-contract "$SABIDS_SPLIT_CONTRACT" --selection-rule "$D1_SELECTION_RULE" \
  "${EVIDENCE_ARGS[@]}" \
  --curves oracle d1 --alphas 0 .25 .5 .75 1 1.25 --seeds 42 43 44 --device cuda
for SEED in 42 43 44; do
  for CURVE in oracle d1; do
    for ALPHA in a000 a025 a050 a075 a100 a125; do
      CFG="cache/adaptive_denoising/$DOSE_PROTOCOL_ID/dose_v1/preparations/full_s42-43-44_full_v1/config_${CURVE}_${ALPHA}_seed${SEED}.yaml"
      python -B train.py --config "$CFG"
    done
  done
done
```

上述命令保留原先声明的六个 alpha，没有从 pilot 挑最佳点；如以后要减少 alpha，
先另行冻结方案/预算，再生成新 tag，不将 pilot 选择混入这次实现。

## 需要从矩池云取得的文件（不猜 D1 run 的路径）

1. **你显式选择的同一 D1 run** 中 `best.pth` 或完整 `last.pth`，以及
   `resolved_config.yaml`, `history.csv`, `run_metadata.json`, `data_plan_audit.json`
   （如果原 run 保存过）, `initialization_audit.json` （如果保存过）。
2. **这一 run 对应的唯一 active protocol 目录**中的
   `active_protocol_lock.json`, `protocol_audit.json`, `train_denoise.csv`,
   `train_segment.csv`, `train_joint.csv`, `label_inventory.csv`, `dataset_inventory.csv`,
   `test_sealed.csv`（只要元数据，不要对应图像/标签）。
3. lock SHA 对应的原 `split_contract.yaml`（实际原文件可能另有名字，明确提供路径，
   不复制一份不同内容的 YAML 充当原文件）。
4. D1 训练时保存的 noisy/clean 像素 inventory；不存在就明确报告缺失。
5. checkpoint 的 denoising manifest 与 `train_segment.csv` 中 **train/val** 行引用的
   noisy/clean/二值 layer/vessel/显式 validity 资产。只取 train/val，不下载 test 图像/标签。

Linux 绝对路径 snapshot 不能在 Windows 上随意改路径后宣称 SHA 仍匹配。
可以在云端原路径执行 formal audit，或单独授权实现可追溯路径映射；本轮没有这种迁移器。

## 本地验证与门槛

```text
python -m compileall -q .
python -m pytest -o addopts='' -q
python -m pytest -o addopts='' -q tests/test_dose_response.py tests/test_dose_geometry.py tests/test_dose_preflight.py
python -B tools/prepare_dose_response_inputs.py --project-root . --mode smoke --synthetic-smoke --budget smoke --curves oracle d1 --alphas 0 .5 --seeds 42 --tag YOUR_FRESH_TAG --device cpu --cpu-smoke
```

CPU smoke 4 合成样本、4 臂（两个 curve × alpha0/.5）、每臂 1 轮，初始状态/数据计划/
实际增强/可训练参数 SHA 配对；不含 test；输出均标记 NOT FOR SCIENTIFIC EVALUATION。
源文件发生变动时应重新跑 smoke，旧缓存拒绝复用是正常保护。

最终门槛必须逐项 PASS/FAIL/BLOCKED：显式 D1 路径、checkpoint SHA、非 smoke、
选择规则、restoration mode、唯一 active protocol、manifest 一致、train/val 无泄漏、
test 未打开、所有臂几何、auxiliary=0、双方向关闭、初始化/顺序/增强配对、单元测试、
CPU smoke、CUDA 短检查。任何正式证据缺失或 CUDA 未验证，不建议开始 pilot。
本地没有正式 D1/active lock 时：IMPLEMENTATION READY, FORMAL PILOT BLOCKED。

### 2026-09-17 首次闭环结果（提交前复审另见下节）

- `python -m compileall -q .`：退出 0。
- 三个新增测试文件：**67 passed, 12 warnings**；全仓库：
  **245 passed, 59 warnings**。警告为弃用 API、CPU pin-memory、缺少 TensorBoard
  （CSV/JSON 仍保存），不是测试失败。
- 最终 CPU 命令中的 fresh tag 为 `minimal_release_v1`：4 合成帧，
  Oracle/D1 两曲线 × alpha0/.5，每臂 1 轮/2 次 optimizer update。
  完整初始化、sampler、actual augmentation、trainable set、cohort 五项 SHA
  配对通过；每臂 114 个可训练参数张量实际改变、冻结参数改变数为 0。
- CPU 缓存/配置/注册表：
  `cache/adaptive_denoising/synthetic_smoke_minimal_release_v1/dose_v1/preparations/smoke_s42_minimal_release_v1/`。
  模型/曲线/元数据：
  `runs/adaptive_denoising/synthetic_smoke_minimal_release_v1/dose_v1/smoke_s42_minimal_release_v1/`。
- 实际 `evaluate.py` CLI 已用 Oracle alpha0 的 `last.pth` 完成全部 2 帧 val，
  输出于该 run 的 `validation_cli_smoke_v1/`。原始 24x40、模型 32x48、
  去 pad 后评价 29x48；文件记录 `cropped_model_grid_px`，
  `scientific_evaluation=false`，不是原图指标或正式性能。
- 最终正式审计：`reports/adaptive_denoising/preflight_local_release_v1/`，
  native exit=2、status=blocked。无显式正式 D1、无 active lock；98 个候选
  checkpoint 均位于 smoke/pilot 路径。候选只列出不选择。
  legacy manifest 8/2/3 位置、368/100/149 帧，未拿来代替历史 13/3、588/141。
  `test_assets_opened=0`。

状态表中的 PASS 表示机制及合成闭环已验证；真实 cohort/D1 的门禁另行 BLOCKED，
不能凭单元测试将缺失正式证据标为 PASS。

| pilot 门槛 | 当前状态 | 证据/限制 |
| --- | --- | --- |
| 明确的 D1 checkpoint 路径 | BLOCKED | 本地未提供正式路径 |
| checkpoint SHA256 已记录 | BLOCKED | smoke SHA 已记录，但不是正式 D1 SHA |
| checkpoint 不是 smoke 权重 | BLOCKED | 没有正式候选 |
| checkpoint 选择规则已冻结 | BLOCKED | 模板支持两种明确规则，未绑定正式模型 |
| restoration mode 已明确 | BLOCKED | 未读取正式 D1 config |
| 唯一 active protocol 已锁定 | BLOCKED | 本地 active lock 数为 0 |
| manifest 与 protocol 一致 | BLOCKED | 两套 cohort 不自动互换 |
| train/val 位置无泄漏 | BLOCKED | 正式 cohort 缺失；合成/拒绝泄漏测试通过 |
| test 资产未打开 | PASS | preflight/smoke 记录 0，sealed guard 测试通过 |
| 所有输入臂几何一致 | BLOCKED | 非方形合成测试通过；正式资产仍待验证 |
| auxiliary loss 为 0 | PASS | resolved config 与 poisoned-output 梯度测试 |
| D→S 和 S→D 关闭 | PASS | alias/legacy 均关闭，交互参数无梯度/更新 |
| 初始化可配对 | PASS | 单元测试 12 臂、CPU 实跑 4 臂 |
| 数据顺序可配对 | PASS | 独立 sampler RNG 与逐 epoch 计划 SHA |
| 增强计划可配对 | PASS | seed/epoch/sample_id 翻转计划 SHA |
| 单元测试通过 | PASS | 最终复审专项 77，全仓库 255 |
| CPU smoke 通过 | PASS | 4 臂、1 轮、有限梯度、checkpoint/metadata |
| CUDA 短 overfit 尚待执行或已通过 | BLOCKED | 本地无 CUDA，本轮只提供命令 |

**IMPLEMENTATION READY, FORMAL PILOT BLOCKED**。没有启动 pilot、多 seed 正式训练、
test 评价或云端操作。首次实现阶段未提交/推送 Git；后续授权仅允许本地功能分支提交，
仍不推送、不更新云端。当前结果仅证明工程链路，不支持任何剂量收益结论。

### 授权提交前最终复审

发现并修正了同进程的全局状态泄漏：剂量 preparation 与 fit 现在使用 scoped
deterministic-algorithm context，成功/异常都恢复原 enabled/warn-only 标志；
旧 Trainer 初始化与旧 fit 不启用剂量专用标志。新增 10 个恢复/关闭路径用例。

修正后真实结果：专项 **77 passed, 12 warnings**，全仓库 **255 passed,
59 warnings**，compileall 与 diff-check 通过。七个核心文件与 base HEAD 的旧路径
对照通过，包括弱/强增强与 RNG、repeat、采样、损失及梯度、全任务 P0--P3 和
原图恢复评价、CLI 默认 test split。新 validation-only/P0/model-grid 限制不改变旧 CLI。

这次只运行审查和用户要求的测试；没有独立启动训练。pytest 中的临时合成 CPU
闭环仍通过。上列 standalone smoke runtime 是修正前源码指纹的历史快照，不回写；
修改源码后若重新生成缓存，应使用 fresh tag，不复用旧 registration。

本地提交分支限定为 `feature/adaptive-denoising-dose-v1`，显式暂存预期 20 文件，
不包括缓存、权重、预测、运行报告或合成数据。新增内容没有绝对 Windows 路径；
`PROJECT_CONTEXT.md` 原有的项目根目录示例没有修改。`.gitignore` 新规则仅锚定
运行产物目录，预期配置/测试/文档均未被忽略。提交信息为
`feat: add gated denoising dose-response experiments`；不自动 push、merge 或更新云端。
