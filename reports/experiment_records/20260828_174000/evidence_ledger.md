# Evidence ledger

| 结论 | 证据 | 限制/反例 | 状态 |
|---|---|---|---|
| 固定PKU validation样本降噪有效 | V0三组 PSNR noisy→denoised、EPI gain均为正 | 仅f01三帧，Stage 2冻结去噪功能 | validation_supported_limited |
| outside BCE主要抑制层外FP | E3b vs E3-current数据身份匹配，E3b full Dice/Precision更高而ROI Dice接近 | 单seed、history为512模型坐标 | validation_supported_limited |
| D→S稳定有效 | E3b与no-D2S hard Dice接近，soft Dice差异很小 | 无重复seed，E1是多因素基线 | not_supported |
| P1/P2改善层mask | 同三帧P0→P1→P2 group-macro Dice上升 | P2同时重建层带并平滑，作用未拆分 | validation_supported_limited |
| P3改善血管 | raw→P3 Dice上升；删除FP=3749、TP=101 | 阈值不是完整P3流程独立校准；只3帧 | validation_supported_limited |
| RMAC/Joint优于Stage 2 | 当前包无完成Joint验证 | 只有历史探索且协议不严谨 | no_evidence |
