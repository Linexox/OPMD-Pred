"""Check complete-expert routing, real H0 capture, and serialized predictions."""
import argparse
from pathlib import Path
import tempfile

import torch

from opmd_pred.model.full_model_moe import FullModelRouter, full_model_features
from opmd_pred.model.two_branch_fusion import TwoBranchFusion
from sysu_binary_common import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--experts', type=int, choices=[4, 5], default=5)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    count = args.experts
    experts = [TwoBranchFusion(512, 14, 2).eval().requires_grad_(False) for _ in range(count)]
    batch = {'image': torch.randn(3, 512), 'table': torch.randn(3, 14),
             'image_present': torch.tensor([True, False, True]),
             'table_present': torch.tensor([True, True, False])}
    h, p = zip(*(full_model_features(model, batch) for model in experts))
    h, p = torch.stack(h, 1), torch.stack(p, 1)
    assert h.shape == (3, count, 3, 128) and p.shape == (3, count, 2)
    for mode in ('mean', 'constant', 'dynamic'):
        model = FullModelRouter(experts=count, mode=mode)
        out, a = model(h, p)
        torch.testing.assert_close(out, p.mean(1))
        torch.testing.assert_close(a, torch.full((3, count), 1 / count))
    dynamic = FullModelRouter(experts=count)
    initial = [v.clone() for model in experts for v in model.parameters()]
    optimizer = torch.optim.AdamW(dynamic.parameters(), lr=.001)
    for _ in range(3):
        mixed, a = dynamic(h, p)
        loss = -mixed[range(3), torch.tensor([0, 1, 1])].log().mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    assert all(param.grad is None for model in experts for param in model.parameters())
    assert all(torch.equal(before, after) for before, after in zip(initial, (v for model in experts for v in model.parameters())))
    mixed, a = dynamic(h, p)
    torch.testing.assert_close(a.sum(1), torch.ones(3))
    torch.testing.assert_close(mixed.sum(1), torch.ones(3))
    assert torch.all(mixed >= p.min(1).values - 1e-7)
    assert torch.all(mixed <= p.max(1).values + 1e-7)
    identical = p[:, :1].expand(-1, count, -1)
    torch.testing.assert_close(dynamic(h, identical)[0], identical[:, 0])
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Temporary round-trip file is a rebuildable check artifact on the project drive.
    with tempfile.TemporaryDirectory(dir=target.parent) as directory:
        path = Path(directory) / 'router.pt'
        torch.save(dynamic.state_dict(), path)
        restored = FullModelRouter(experts=count)
        restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
        torch.testing.assert_close(restored(h, p)[0], mixed, rtol=0, atol=0)
    write(target, {'passed': True, 'h0_shape': list(h.shape), 'complete_expert_probability_shape': list(p.shape),
        'zero_initialization_is_mean': True, 'expert_parameters_unchanged': True,
        'weights_sum_to_one': True, 'convex_combination_bounds': True, 'reload_exact': True,
        'dynamic_trainable_parameters': sum(x.numel() for x in dynamic.parameters())})
    print('Complete-model Router checks passed', flush=True)


if __name__ == '__main__':
    main()
