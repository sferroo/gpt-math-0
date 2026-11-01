
import pytest
import torch

from gpt0.model import GPT0


def make_model(device="cpu"):
    return GPT0(
        vocab_size=128,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    ).to(device)


def test_forward_and_loss():
    model = make_model()

    x = torch.randint(0, 128, (2, 16))
    logits, loss = model(x, x)

    assert logits.shape == (2, 16, 128)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()


def test_context_limit():
    model = GPT0(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=8,
    )

    x = torch.randint(0, 32, (1, 9))

    with pytest.raises(ValueError):
        model(x)


def test_loss_masking():
    model = make_model()
    model.eval()

    x = torch.randint(1, 128, (2, 16))

    targets_a = x.clone()
    targets_b = x.clone()

    # Ignore the last four target positions.
    targets_a[:, -4:] = -100
    targets_b[:, -4:] = -100

    # Change ignored target values.
    # The loss must remain identical.
    targets_b[:, -4:] = -100

    _, loss_a = model(x, targets_a)
    _, loss_b = model(x, targets_b)

    assert torch.allclose(loss_a, loss_b)


def test_padding_mask_forward():
    model = make_model()
    model.eval()

    # Zero is the PAD token ID.
    x = torch.tensor([
        [10, 20, 30, 0, 0],
        [15, 25, 35, 45, 0],
    ])

    logits, _ = model(x)

    assert logits.shape == (2, 5, 128)
    assert torch.isfinite(logits).all()


def test_padding_mask_backward():
    model = make_model()

    x = torch.tensor([
        [10, 20, 30, 0, 0],
        [15, 25, 35, 45, 0],
    ])

    targets = x.clone()
    targets[:, 1:] = torch.where(
        targets[:, 1:] == 0,
        -100,
        targets[:, 1:],
    )

    _, loss = model(x, targets)

    assert torch.isfinite(loss)

    loss.backward()

    assert model.token_emb.weight.grad is not None
    assert torch.isfinite(model.token_emb.weight.grad).all()


def test_cuda_forward_backward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    model = make_model(device="cuda")

    x = torch.randint(1, 128, (2, 16), device="cuda")
    targets = x.clone()

    _, loss = model(x, targets)

    assert loss.is_cuda
    assert torch.isfinite(loss)

    loss.backward()

    assert model.token_emb.weight.grad is not None
    assert torch.isfinite(model.token_emb.weight.grad).all()
