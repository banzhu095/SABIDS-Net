# I-NOISY / I-DENOISED / I-CLEAN audit

This report audits fixed-final checkpoints and does not select a checkpoint or threshold.

## Interpretation

> 当前实验是“相同 Stage 1 预训练初始化后，针对三种输入分别进行匹配微调”，不是整个网络从随机权重训练，也不是完全冻结编码器的输入探针。

`stage2_freeze_shared_encoder: true` conflicts with the actual trainable `stem/encoder_blocks/downsamples` in 3 run(s).

## Fold pairing

```json
{
  "0": {
    "checks": {
      "all_three_arms_present": true,
      "same_initialization_sha256": true,
      "same_model_state_sha256": true,
      "same_common_state_sha256": true,
      "same_data_plan_sha256": true,
      "same_manifest_sha256": true,
      "same_training_positions": true,
      "same_validation_positions": true,
      "same_augmentation_plan": true,
      "same_configured_epoch": true,
      "all_epoch_60": true,
      "all_d2s_off": true,
      "all_s2d_off": true,
      "all_denoising_frozen": true,
      "all_interactions_frozen": true
    },
    "paired_protocol_pass": true
  }
}
```
