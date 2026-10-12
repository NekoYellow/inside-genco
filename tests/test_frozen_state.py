"""Protect checkpoint state while allowing non-persistent observation caches."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from baseline import model_state_sha256


class FrozenStateTests(unittest.TestCase):
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
