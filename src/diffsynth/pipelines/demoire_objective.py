"""The owner-selected objective (manuscript plus latent MSE); LPIPS uses its own VGG network, not a separate feature loss."""
import torch

def endpoint_objective(velocity_loss, prediction, target, latent_prediction, latent_target, lpips_fn, ffl_fn,
                       velocity=1.0, latent_mse=1.0, l1=0.5, lpips_weight=0.3, ffl_weight=0.2):
    if prediction.shape != target.shape:
        raise ValueError('Prediction and clean target shapes must match')
    if latent_prediction.shape != latent_target.shape:
        raise ValueError('Latent prediction and clean target shapes must match')
    if lpips_fn is None or ffl_fn is None:
        raise RuntimeError('LPIPS and FFL must both be available')
    terms = {
        'velocity_loss': velocity_loss,
        'latent_mse_loss': torch.nn.functional.mse_loss(latent_prediction.float(), latent_target.float()),
        'l1_loss': torch.nn.functional.l1_loss(prediction.float(), target.float()),
        'lpips_loss': lpips_fn(prediction.float(), target.float()).mean(),
        'ffl_loss': ffl_fn(prediction.float(), target.float()),
    }
    for name, value in terms.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite objective component: ' + name)
    total = velocity * terms['velocity_loss'] + latent_mse * terms['latent_mse_loss'] + l1 * terms['l1_loss'] + lpips_weight * terms['lpips_loss'] + ffl_weight * terms['ffl_loss']
    terms['total_loss'] = total
    return total, terms
