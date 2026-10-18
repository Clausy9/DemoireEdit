"""CPU checks of weighting and gradients; LPIPS/FFL networks are test doubles."""
import importlib.util
from pathlib import Path
import unittest
try:
    import torch
except ImportError:
    torch = None

@unittest.skipIf(torch is None, 'PyTorch is required for objective checks')
class ObjectiveChecks(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'src/diffsynth/pipelines/demoire_objective.py'
        spec = importlib.util.spec_from_file_location('objective_under_test',path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.objective = module.endpoint_objective
        self.lpips = lambda a,b: (a-b).square().mean()
        self.ffl = lambda a,b: (a-b).abs().mean()
    def test_latent_mse_contributes_and_backpropagates(self):
        prediction=torch.ones(1,3,2,2,requires_grad=True)
        latent=torch.full((1,4,1,1),2.,requires_grad=True)
        total,terms=self.objective(torch.tensor(3.),prediction,torch.zeros_like(prediction),latent,torch.zeros_like(latent),self.lpips,self.ffl)
        self.assertEqual(set(terms),{'velocity_loss','latent_mse_loss','l1_loss','lpips_loss','ffl_loss','total_loss'})
        self.assertAlmostEqual(total.item(),8.,places=5)
        total.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertGreater(latent.grad.abs().sum().item(),0)
    def test_nonfinite_terms_stop(self):
        p=torch.ones(1,3,2,2);z=torch.zeros(1,4,1,1)
        with self.assertRaises(FloatingPointError):
            self.objective(torch.tensor(float('nan')),p,p,z,z,self.lpips,self.ffl)
    def test_missing_loss_network_stops(self):
        p=torch.ones(1,3,2,2);z=torch.zeros(1,4,1,1)
        with self.assertRaises(RuntimeError):
            self.objective(torch.tensor(0.),p,p,z,z,None,self.ffl)
    def test_latent_shape_mismatch_stops(self):
        p=torch.ones(1,3,2,2);z=torch.zeros(1,4,1,1)
        with self.assertRaises(ValueError):
            self.objective(torch.tensor(0.),p,p,z,torch.zeros(1,4,2,2),self.lpips,self.ffl)
if __name__=='__main__': unittest.main()
