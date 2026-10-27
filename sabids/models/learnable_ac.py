"""Opt-in, bounded task-specific AC controls; no GT is a forward argument.

The analytic candidate is ac_r2's *already reliable* safe_output. rho below
only moderates additional adjustments; it does not apply base reliability twice.
"""
from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F

from sabids.experiments.attenuation_aux import StableACConfig


def fixed_linear_resize(value, size):
    """align_corners=False linear interpolation with deterministic CUDA VJP.

    PyTorch2.1 CUDA's upsample_linear1d backward rejects deterministic mode.
    This fixed (nonlearnable) interpolation matrix has the same half-pixel
    grid and edge replication; its backward is a deterministic GEMM.
    """
    length=value.shape[-1]
    if length==size:return value
    grid=((torch.arange(size,device=value.device,dtype=torch.float64)+.5)*(length/size)-.5).clamp(0,length-1)
    source=torch.arange(length,device=value.device,dtype=torch.float64)
    weights=(1-(source[:,None]-grid[None,:]).abs()).clamp_min(0).to(value.dtype)
    return value @ weights


class NativeStableAC(nn.Module):
    """Float64 native A-line integration, valid-neighbour trapezoids only.

    Accept Bx1xHxW, with contiguous valid depth or explicit invalid gaps. The
    spatially transformed tensors must never be passed to this entry instead
    of the native image. Output c/z0 are float32 just like the historical cache.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config if isinstance(config, StableACConfig) else StableACConfig(**config)
        self.config.validate()
        if self.config.beta != .5:
            raise ValueError('learnable_ac_v1 requires fixed base beta=.5')

    def forward(self, image, valid=None):
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError('Expected native Bx1xHxW')
        if not torch.isfinite(image).all() or (image < 0).any() or (image > 1).any():
            raise ValueError('Expected finite unit intensity')
        mask = torch.ones_like(image, dtype=torch.bool) if valid is None else valid.bool()
        if mask.shape != image.shape:
            raise ValueError('Native validity shape mismatch')
        cfg = self.config
        x = image.double()
        energy = x.pow(cfg.power) * mask
        area = .5 * (energy[..., :-1, :] + energy[..., 1:, :]) * cfg.depth_spacing
        area = area * (mask[..., :-1, :] & mask[..., 1:, :])
        integral = F.pad(area.flip(-2).cumsum(-2).flip(-2), (0, 0, 0, 1))
        n = mask.sum(-2, keepdim=True).double()
        raw = n * energy / (integral + cfg.denominator_floor)
        reliability = integral / (integral + cfg.denominator_floor)
        weight = ((reliability - cfg.reliability_floor) / (1-cfg.reliability_floor)).clamp(0, 1)
        unmapped = raw / cfg.output_window
        mapped = unmapped.clamp(0, 1)
        capped = torch.minimum(mapped, x * cfg.gain_cap)
        safe = torch.where(mask, weight*capped + (1-weight)*x, x)
        # Mix in float64 before casting to match numpy r2's rounding exactly.
        z0 = (.5*x + .5*safe).clamp(0, 1).float()
        gain = torch.where(x > 0, capped/x.clamp_min(1e-300), torch.ones_like(x))
        # Geometric depth does not use GT or predicted layer thickness.
        rank = mask.cumsum(-2).double()-1
        depth = torch.where(mask, rank/(n-1).clamp_min(1), torch.zeros_like(rank))
        # Extra-adjustment reliability: energy support and saturation, NOT gray
        # darkness. Dark vascular lumen is not rejected. c is already protected.
        rho = mask.double() * reliability * (1-.5*(unmapped >= 1).double())
        return dict(m=image.float(), c=safe.float(), z0=z0, energy=integral.float(),
                    reliability=reliability.float(), gain=gain.float(), depth=depth.float(),
                    rho=rho.float(), valid=mask.float(),
                    denominator_active=((integral < cfg.denominator_floor)&mask).float(),
                    gain_cap_active=((mapped > x*cfg.gain_cap)&mask).float(),
                    high_clip=((unmapped >= 1)&mask).float(),
                    low_clip=((unmapped <= 0)&mask).float(),
                    fallback=((weight <= 0)&mask).float())

    def metadata(self):
        return dict(version='learnable_ac_v1.native_r2', config=asdict(self.config),
                    candidate='safe_output after base reliability, BEFORE beta mix',
                    rho='valid * integral/(integral+regularizer) * (1-.5*high_clip)',
                    trainable_ac_parameters=[], output_domain='unit intensity')


class ContextEncoder(nn.Module):
    def __init__(self, channels=(16,32,64)):
        super().__init__()
        blocks=[]; previous=10
        for c in channels:
            blocks.append(nn.Sequential(nn.Conv2d(previous,c,3,stride=2,padding=1),
                nn.GroupNorm(4,c),nn.SiLU(),nn.Conv2d(c,c,3,padding=1),nn.GroupNorm(4,c),nn.SiLU()))
            previous=c
        self.blocks=nn.ModuleList(blocks)

    def forward(self,x):
        for block in self.blocks:x=block(x)
        return x


def entropy(p):
    p=p.clamp(1e-6,1-1e-6)
    return -(p*p.log()+(1-p)*(1-p).log()) / .6931471805599453


class ACController(nn.Module):
    """Four smooth depth bases and low-resolution lateral coefficients."""
    def __init__(self, method='F5', delta=.25, channels=(16,32,64), bases=4):
        super().__init__()
        if method not in ('F1','F2','F3','F4','F5','F6') or delta not in (.25,.5):
            raise ValueError('Unknown method or unregistered beta range')
        self.method,self.delta,self.bases=method,delta,bases
        if method=='F2': self.global_raw=nn.Parameter(torch.zeros(2))
        else:
            self.context=ContextEncoder(channels)
            self.head=nn.Conv1d(channels[-1], bases*(1 if method=='F3' else 2),1)
            nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,context,depth,rho):
        b,_,h,w=context.shape
        if self.method=='F6':
            context=torch.cat([context[:,:6],torch.zeros_like(context[:,6:])],1)
        if self.method=='F2':
            coeff=self.global_raw[None,:,None,None].expand(b,2,1,w)
            raw=coeff.squeeze(2)[:,:,None,:].expand(b,2,h,w)
        else:
            lateral=self.context(context).mean(-2)
            coeff=self.head(lateral).reshape(b,-1,self.bases,lateral.shape[-1])
            coeff=fixed_linear_resize(coeff.reshape(b,-1,lateral.shape[-1]),w)
            coeff=coeff.reshape(b,-1,self.bases,w)
            if self.method=='F3':coeff=coeff.expand(-1,2,-1,-1)
            centers=torch.linspace(0,1,self.bases,device=depth.device,dtype=depth.dtype)
            basis=torch.exp(-.5*((depth-centers[None,:,None,None])/.25)**2)
            basis=basis/basis.sum(1,keepdim=True).clamp_min(1e-6)
            raw=torch.einsum('btkx,bkhx->bthx',coeff,basis)
        delta_beta=self.delta*raw.tanh()
        # F2 is genuinely two GLOBAL beta values. Spatial rho would turn it
        # into a varying beta and invalidate that control. c remains protected.
        r=torch.ones_like(rho) if self.method in ('F2','F4') else rho
        adjustment=delta_beta*r
        return dict(coefficients=coeff,delta_beta=delta_beta,rho=r,
                    effective_beta=.5+adjustment,adjustment=adjustment)


def task_forward(segmenter, image, task):
    features=segmenter.encode(image)
    x=segmenter.adapters[task][-1](features[-1])
    for j,i in enumerate(segmenter.levels):
        x=segmenter.decoders[task][j](x,segmenter.adapters[task][i](features[i]))
    return segmenter.heads[task](x)


class AdaptiveSegmenter(nn.Module):
    """Frozen S0; refined forwards intentionally retain input autograd."""
    def __init__(self, anchor, method='F5',delta=.25,channels=(16,32,64)):
        super().__init__()
        self.anchor=anchor.eval().requires_grad_(False)
        self.method=method
        self.control=None if method=='F0' else ACController(method,delta,channels)
        self.last_diagnostics={}

    def train(self,mode=True):
        super().train(mode);self.anchor.eval()
        return self

    def forward(self,packed,adaptive_off=False):
        # packed image-only channels: m,z0,c,depth,reliability,gain,rho,valid.
        if packed.ndim!=4 or packed.shape[1]!=8:
            raise ValueError('Expected eight image-only analytic channels')
        m,z0,c,depth,reliability,gain,rho,valid=packed.split(1,1)
        with torch.no_grad():coarse=self.anchor(z0)
        self.last_diagnostics=dict(coarse_layer=coarse['layer'].sigmoid(),
            coarse_vessel=coarse['vessel'].sigmoid(),layer_image=z0,vessel_image=z0,
            effective_beta=torch.full_like(torch.cat([z0,z0],1),.5),rho=rho,
            delta_beta=torch.zeros_like(torch.cat([z0,z0],1)),valid=valid,encoder_calls=1)
        if adaptive_off or self.control is None:return coarse
        pl,pv=coarse['layer'].sigmoid(),coarse['vessel'].sigmoid()
        context=torch.cat([m,z0,c-m,depth,reliability,gain/6,pl,pv,entropy(pl),entropy(pv)],1).detach()
        diag=self.control(context,depth,rho*valid)
        adjustment=diag['adjustment']*valid
        if self.method=='F1':
            # Matched context + identical head parameter count; low-rank logit
            # refiner, no AC image alteration. Same two task forwards as F5.
            zl,zv=z0,z0
            out={t:task_forward(self.anchor,z0,t)+4*adjustment[:,i:i+1]
                 for i,t in enumerate(('layer','vessel'))}
        else:
            zl=(z0+adjustment[:,0:1]*(c-m)).clamp(0,1)
            zv=(z0+adjustment[:,1:2]*(c-m)).clamp(0,1)
            out=dict(layer=task_forward(self.anchor,zl,'layer'),vessel=task_forward(self.anchor,zv,'vessel'))
        self.last_diagnostics.update(diag,layer_image=zl,vessel_image=zv,encoder_calls=3)
        return out


def control_regularization(diagnostics):
    d=diagnostics['delta_beta']
    valid=diagnostics.get('valid',torch.ones_like(d[:,:1]))
    zero=d.sum()*0
    def mean(value,mask):
        mask=mask.expand_as(value)
        return (value*mask).sum()/mask.sum().clamp_min(1)
    depth=mean((d[...,2:,:]-2*d[...,1:-1,:]+d[...,:-2,:]).square(),
        valid[...,2:,:]*valid[...,1:-1,:]*valid[...,:-2,:]) if d.shape[-2]>2 else zero
    lateral=mean((d[...,1:]-d[...,:-1]).square(),valid[...,1:]*valid[...,:-1]) if d.shape[-1]>1 else zero
    return dict(depth_smooth=depth,lateral_smooth=lateral,anchor=mean(d.square(),valid))
