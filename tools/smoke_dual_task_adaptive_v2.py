from __future__ import annotations
import json, sys, tempfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from sabids.models.dual_task_adaptive_v2 import DualTaskAdaptiveV2Segmenter
from sabids.experiments.dual_task_adaptive_v2 import vessel_protection_losses

def model():
 kw=dict(in_channels=1,channels=(4,8,16,32),encoder_depths=(1,1,1,1),decoder_depth=1,
 interaction_levels=(3,2,1),enable_seg_to_denoise=False,enable_denoise_to_seg=False,
 use_uncertainty=False,detach_denoise_to_seg_source=True,dropout=0.,residual_scale=.5,
 causal_interaction_experiment=False,detach_seg_to_denoise_source=True,interaction_scale_init=.1,
 s2d_source_mode='cross',d2s_source_mode='cross',strong_s2d_rho=None,strong_d2s_rho=None)
 return DualTaskAdaptiveV2Segmenter(kw,context_channels=4)
def main():
 torch.manual_seed(42); m=model(); x=torch.rand(1,1,24,32); vessel=torch.zeros_like(x); vessel[:,:,5:12,7:15]=1; layer=torch.ones_like(x); valid=torch.ones_like(x)
 before={n:p.detach().clone() for n,p in m.named_parameters()}; opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3)
 losses=[]
 for _ in range(3):
  out=m(x); segmentation=torch.nn.functional.binary_cross_entropy_with_logits(out['vessel_logits'],vessel)
  protect=vessel_protection_losses(out['vessel_strength_map'],vessel,layer,valid,vessel)['vessel_protect']
  loss=segmentation+.05*protect; opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
 changed_trainable=[n for n,p in m.named_parameters() if p.requires_grad and not torch.equal(before[n],p)]
 changed_frozen=[n for n,p in m.named_parameters() if not p.requires_grad and not torch.equal(before[n],p)]
 off=m(x,vessel_adaptive_off=True)
 result={'status':'passed' if changed_trainable and not changed_frozen and torch.equal(off['vessel_logits'],off['coarse_vessel_logits']) else 'failed',
 'shape':list(off['vessel_logits'].shape),'losses':losses,'changed_trainable_parameter_names':changed_trainable,
 'changed_frozen_parameter_names':changed_frozen,'off_exact_coarse':bool(torch.equal(off['vessel_logits'],off['coarse_vessel_logits'])),
 'test_assets_opened':0,'notice':'synthetic CPU smoke; not scientific evidence'}
 print(json.dumps(result,indent=2)); raise SystemExit(0 if result['status']=='passed' else 1)
if __name__=='__main__': main()
