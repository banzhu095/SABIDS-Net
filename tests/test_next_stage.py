import torch
from sabids.losses.common import multiscale_gradient_loss, multiscale_laplacian_loss
from sabids.models.ugbi import UGBIBlock

def test_d1_structure_losses_finite_backward():
    x=torch.rand(1,1,32,32,requires_grad=True); y=torch.rand_like(x)
    loss=multiscale_gradient_loss(x,y)+multiscale_laplacian_loss(x,y)
    assert torch.isfinite(loss); loss.backward(); assert torch.isfinite(x.grad).all()

def test_rms_update_and_zero_equivalence():
    update=torch.randn(2,4,8,8,requires_grad=True); receiver=torch.randn_like(update)
    assert torch.equal(UGBIBlock.rms_scaled_update(update,receiver,0),torch.zeros_like(update))
    scaled=UGBIBlock.rms_scaled_update(update,receiver,0.01)
    ratio=scaled.square().mean((1,2,3)).sqrt()/receiver.square().mean((1,2,3)).sqrt()
    assert torch.allclose(ratio,torch.full_like(ratio,0.01),atol=1e-5)
    scaled.sum().backward(); assert update.grad is not None

def test_detached_source_blocks_source_gradient_but_mapping_trains():
    block=UGBIBlock(4,enable_seg_to_denoise=False,enable_denoise_to_seg=True)
    source=torch.randn(1,4,8,8,requires_grad=True); layer=torch.randn_like(source,requires_grad=True); vessel=torch.randn_like(source,requires_grad=True)
    lo,vo,_=block.denoise_to_seg(source,layer,vessel,detach_source=True,rms_rho=.01)
    (lo.mean()+vo.mean()).backward()
    assert source.grad is None and layer.grad is not None
    assert block.denoise_to_layer.weight.grad is not None
