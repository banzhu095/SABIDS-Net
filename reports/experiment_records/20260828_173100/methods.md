# 当前真实方法与证据状态

## 网络图

```text
noisy B-scan
  └─ shared stem + four-scale encoder
       ├─ task adapters → denoise decoder → residual head → noisy - 0.5*tanh(residual)
       ├─ task adapters → layer decoder   → layer + two-channel boundary heads
       └─ task adapters → vessel decoder  → vessel head
             ↕ UGBI at levels 3/2/1 (only enabled directions act)
```

分割不是对 `denoised` 图像执行第二次显式 forward；三条路径共享 noisy 编码特征，D→S 通过 UGBI 将 restoration context 注入层/血管特征。源码：`sabids/models/sabids_net.py`、`sabids/models/ugbi.py`。

| 条目 | proposed | implemented | run_evidence | validation_supported | 说明/源码 |
|---|---|---|---|---|---|
| 共享编码器、三任务适配器/解码器 | yes | yes | yes | Stage 2 only | `sabids/models/sabids_net.py` |
| 残差降噪 | yes | yes | Stage 1初始化及V0输出 | 固定3组支持 | `denoised_raw=image-residual`，展示输出clamp |
| D→S UGBI | yes | yes | E3b及完整no-D2S训练 | 未显示稳定优势 | no-D2S hard Dice接近E3b；需seed/完整配对评价 |
| S→D UGBI | yes | yes | 当前四Stage 2关闭 | no | 不确定性置信主要作用于S→D anatomy融合，当前结果不能证明其有效 |
| E3b ROI BCE+Dice | yes | yes | yes | 当前validation支持 | GT layer ROI内逐图BCE+Dice |
| outside BCE | yes | yes | E3b/E3-current单因素 | 支持抑制层外FP | FP32 softplus(logit)，空outside跳过 |
| containment | yes | yes | yes | 与outside共同存在 | 有GT层用GT，否则detach预测层；不能单独防整层血管 |
| layer boundary loss | yes | yes | yes | 有layer指标 | boundary_weight=0.2为层loss内部边界BCE权重，不是独立总loss权重 |
| RMAC | yes | yes | 当前Stage 2权重为0 | no completed Joint evidence | `sabids/losses/rmac.py` |
| memory-safe repeat/clean stop-gradient | yes | yes | 仅smoke/未来Joint | no | `sabids/engine/trainer.py` |
| EMA+暗腔双源伪标签 | yes | yes | Stage 5尚无正式结果 | no | `sabids/losses/pseudo.py` |
| P1/P2/P3后处理 | later plan | yes | V0 3固定帧 | limited | P3严格相交；越界=0是构造性质 |

## Stage 2 冻结与梯度

E3b/E3-current冻结去噪分支和完整共享上游，S→D关闭；D→S来源被detach，仅训练接收侧交互和分割路径。no-D2S关闭该交互。E1-current是复合监督整体基线，不是单因素loss对照。Stage 2 reconstruction、RMAC、pseudo均不在active集合中。

## 损失与评价边界

Stage 1 reconstruction = Charbonnier + 0.2 MS-SSIM loss + 0.1 wavelet + 0.1边缘项，另有residual L1。E3b分割主目标为layer、ROI vessel、outside与containment；不同loss定义的total不能横向排名。阈值校准在512模型坐标，而V0恢复到640原图坐标；二者不可视为已完全复现的一致评价。
