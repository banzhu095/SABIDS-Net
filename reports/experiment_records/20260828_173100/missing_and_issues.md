# Missing and issues

## P0

1. V0只有每组f01，共3帧；manifest原始组帧为50/50/41，尚无完整141帧评价。
2. calibration在512模型坐标，V0在640恢复坐标；raw最佳0.425与V0 raw阈值0.35不可直接核对。
3. component bins `[126,301]`在512坐标推导，却用于640 V0；面积阈值不可直接跨坐标沿用，小/中/大Recall标为定义不匹配。
4. P3沿用raw阈值，未对最终P3流程单独校准。
5. patient_id在V0等于group_id，只能确认group隔离，真实患者身份未核验。

## P1

6. 归档包只含E3b的V0预测，无法对四模型做同帧、同坐标定性比较。
7. clean PNG未导出，图册只能展示noisy/denoised；数值来自已有浮点评价日志。
8. 历史CSV缺少metadata/config/checkpoint/指纹，协议和权重保存状态为unknown。
9. `p1_component_count`是清理前计数；hole为二值填洞，开放凹口不计为hole。
10. P2从列上下边界重建层带并平滑下边界，现有结果不能把填带与平滑贡献拆开。

## 待评价补丁（未执行）

- 全141 validation帧 records-compatible推理，独立eval_id；保持test未触碰。
- 在同一恢复几何下重新校准raw/soft-gate/P3，并在640坐标重新推导component bins。
- 导出浮点概率或无损数组、clean图、完整TP/FP/FN及P3删除TP/FP图。
- 四模型使用同一固定帧集合导出定性结果。
