import copy
import unittest
from pathlib import Path
from demoire_edit.recipe import condition_shift_steps, condition_schedule, load_recipe, validate_recipe
class RecipeChecks(unittest.TestCase):
    def test_exact_selected_switch(self):
        self.assertEqual(condition_schedule(.3, 10), ['stage1_output'] * 3 + ['original_input'] * 7)
    def test_ceiling_and_boundaries(self):
        self.assertEqual(condition_shift_steps(.31, 10), 4)
        self.assertEqual(condition_shift_steps(0, 10), 0)
        self.assertEqual(condition_shift_steps(1, 10), 10)
    def test_invalid_ratio(self):
        for value in [-.1, 1.1, float('nan'), float('inf'), True]:
            with self.assertRaises(ValueError): condition_shift_steps(value, 10)
    def test_latent_mse_is_required(self):
        recipe = load_recipe(Path(__file__).resolve().parents[1] / 'configs/paper.json')
        self.assertEqual(recipe['training']['loss']['latent_mse'], 1.0)
        del recipe['training']['loss']['latent_mse']
        with self.assertRaises(ValueError): validate_recipe(recipe)
    def test_extra_losses_rejected(self):
        recipe = load_recipe(Path(__file__).resolve().parents[1] / 'configs/paper.json')
        for extra in ['vgg', 'lab', 'chroma']:
            bad = copy.deepcopy(recipe); bad['training']['loss'][extra] = .1
            with self.assertRaises(ValueError): validate_recipe(bad)
if __name__ == '__main__': unittest.main()
