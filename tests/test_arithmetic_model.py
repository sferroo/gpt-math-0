
import pytest
import torch

from gpt0.arithmetic_model import ArithmeticGPT
from gpt0.numeric_tokenizer import NumericTokenizer


def make_model():
    return ArithmeticGPT(
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )


def test_model_configuration():
    model = make_model()

    assert model.vocab_size == 15
    assert model.block_size == 32
    assert model.pad_token_id == NumericTokenizer.PAD


def test_forward_shape():
    model = make_model()

    sequences = [
        NumericTokenizer.encode_sequence(78, 45)[:-1],
        NumericTokenizer.encode_sequence(12, 34)[:-1],
    ]

    max_len = max(len(sequence) for sequence in sequences)

    x = torch.tensor([
        sequence + [NumericTokenizer.PAD] * (max_len - len(sequence))
        for sequence in sequences
    ])

    logits, loss = model(x)

    assert logits.shape == (2, x.shape[1], 15)
    assert loss is None


def test_forward_with_masked_loss():
    model = make_model()

    x = torch.tensor([
        [1, 12, 13, 3, 9, 10, 4, 6, 7, 8],
    ])

    labels = torch.tensor([
        [-100, -100, -100, -100, -100, -100, 6, 7, 8, 2],
    ])

    logits, loss = model(x, labels)

    assert logits.shape == (1, 10, 15)
    assert loss is not None
    assert torch.isfinite(loss)


def test_backward_produces_finite_gradients():
    model = make_model()

    x = torch.tensor([
        [1, 12, 13, 3, 9, 10, 4, 6, 7, 8],
    ])

    labels = torch.tensor([
        [-100, -100, -100, -100, -100, -100, 6, 7, 8, 2],
    ])

    _, loss = model(x, labels)

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients
    assert all(torch.isfinite(grad).all() for grad in gradients)


def test_invalid_target_token_rejected():
    model = make_model()

    x = torch.tensor([[1, 12, 3, 4, 6]])

    invalid_targets = torch.tensor([
        [-100, -100, -100, 3, 2],
    ])

    with pytest.raises(ValueError):
        model(x, invalid_targets)


def test_target_shape_mismatch_rejected():
    model = make_model()

    x = torch.tensor([[1, 12, 3, 4]])

    targets = torch.tensor([[2, 3]])

    with pytest.raises(ValueError):
        model(x, targets)


def test_causal_attention_blocks_future_information():
    model = make_model()
    model.eval()

    x1 = torch.tensor([
        [1, 12, 13, 3, 9, 10, 4],
    ])

    x2 = x1.clone()

    # Change a future token.
    x2[0, 6] = 5

    with torch.no_grad():
        logits1, _ = model(x1)
        logits2, _ = model(x2)

    # Earlier positions must not depend on future tokens.
    assert torch.allclose(
        logits1[:, :6, :],
        logits2[:, :6, :],
        atol=1e-6,
    )


def test_generation_requires_eval_mode():
    model = make_model()

    prompt = torch.tensor([
        NumericTokenizer.encode_input(78, 45)
    ])

    with pytest.raises(RuntimeError):
        model.generate(prompt)


def test_generation_produces_only_digits_and_eos():
    model = make_model()
    model.eval()

    prompt = torch.tensor([
        NumericTokenizer.encode_input(78, 45)
    ])

    output = model.generate(
        prompt,
        max_new_tokens=5,
    )

    generated = output[0, prompt.shape[1]:].tolist()

    assert len(generated) >= 1
    assert all(
        token == NumericTokenizer.EOS
        or NumericTokenizer.DIGIT_OFFSET <= token
        < NumericTokenizer.VOCAB_SIZE
        for token in generated
    )


def test_generation_rejects_invalid_prompt():
    model = make_model()
    model.eval()

    invalid_prompt = torch.tensor([[1, 12, 13, 4]])

    with pytest.raises(ValueError):
        model.generate(invalid_prompt)


def test_generation_rejects_empty_generation_budget():
    model = make_model()
    model.eval()

    prompt = torch.tensor([
        NumericTokenizer.encode_input(78, 45)
    ])

    with pytest.raises(ValueError):
        model.generate(prompt, max_new_tokens=0)
