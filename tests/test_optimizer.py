import torch
import torch.nn as nn

from training.optimizer import Muon


def test_muon_linear_step():
    torch.manual_seed(42)

    model = nn.Linear(8, 4, bias=False)

    # Save the initial weights so we can verify they actually change.
    initial_weight = model.weight.detach().clone()

    optimizer = Muon([
        {
            "params": [model.weight],
            "use_muon": True,
            "lr": 0.02,
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 5,
            "weight_decay": 0.0,
        }
    ])

    x = torch.randn(16, 8)
    target = torch.randn(16, 4)

    # Forward pass
    output = model(x)

    # Simple loss
    loss = ((output - target) ** 2).mean()

    # Backward pass
    loss.backward()

    # Make sure the gradient exists before the optimizer touches it.
    assert model.weight.grad is not None

    # Optimizer step
    optimizer._muon_step(optimizer.param_groups[0])

    # The parameter should have changed.
    assert not torch.equal(model.weight, initial_weight)

    # Muon should have created its momentum state.
    state = optimizer.state[model.weight]

    assert "momentum_buffer" in state

    # The momentum buffer should have the same shape as the parameter.
    assert state["momentum_buffer"].shape == model.weight.shape


def test_muon_multiple_steps():
    torch.manual_seed(42)

    model = nn.Linear(8, 4, bias=False)

    optimizer = Muon([
        {
            "params": [model.weight],
            "use_muon": True,
            "lr": 0.02,
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 5,
            "weight_decay": 0.0,
        }
    ])

    x = torch.randn(16, 8)
    target = torch.randn(16, 4)

    for _ in range(3):
        optimizer.zero_grad()

        output = model(x)
        loss = ((output - target) ** 2).mean()

        loss.backward()

        optimizer._muon_step(optimizer.param_groups[0])

    # State should persist across optimizer steps.
    state = optimizer.state[model.weight]

    assert "momentum_buffer" in state
    assert state["momentum_buffer"].shape == model.weight.shape