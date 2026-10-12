"""Protect checkpoint state while allowing non-persistent observation caches."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from baseline import model_state_sha256, physicalize_output


class _FakeBatch:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return _FakeBatch(self.value.clone())

    def apply(self, function):
        self.value = function(self.value)
        return self


class _IdentityNormalizer:
    def inverse_transform(self, batch):
        batch.value.add_(1)

    def inverse_output(self, output, batch):
        output["bus"].add_(batch.value)


class FrozenStateTests(unittest.TestCase):
    def test_physicalize_output_clones_and_honors_metric_dtype(self):
        output = {"bus": torch.tensor([2.0], dtype=torch.float32)}
        batch = _FakeBatch(torch.tensor([3.0], dtype=torch.float32))

        physical_output, physical = physicalize_output(
            output, batch, _IdentityNormalizer(), "float64"
        )

        self.assertEqual(physical_output["bus"].dtype, torch.float64)
        self.assertEqual(physical.value.dtype, torch.float64)
        torch.testing.assert_close(
            physical_output["bus"], torch.tensor([6.0], dtype=torch.float64)
        )
        torch.testing.assert_close(output["bus"], torch.tensor([2.0]))
        torch.testing.assert_close(batch.value, torch.tensor([3.0]))

    def test_persistent_state_changes_are_detected(self):
        model=torch.nn.Linear(2,1).eval().requires_grad_(False)
        model.register_buffer('counter',torch.tensor(0,dtype=torch.long))
        expected=model_state_sha256(model)
        model.layer_residuals={0:torch.tensor(42.)}
        self.assertEqual(expected,model_state_sha256(model))
        model.counter+=1
        self.assertNotEqual(expected,model_state_sha256(model))
        model.counter-=1
        self.assertEqual(expected,model_state_sha256(model))
        model.weight[0,0]+=1
        self.assertNotEqual(expected,model_state_sha256(model))


if __name__=='__main__':unittest.main()
