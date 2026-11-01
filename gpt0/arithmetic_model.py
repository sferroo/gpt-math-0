from __future__ import annotations

import torch

from gpt0.model import GPT0
from gpt0.numeric_tokenizer import NumericTokenizer


class ArithmeticGPT(GPT0):
    """
    GPT0 specialized for autoregressive decimal addition.

    Keeps the original Transformer architecture while using:
    - A 15-token numeric vocabulary
    - A shorter context window
    - Constrained autoregressive numeric generation
    """

    def __init__(
        self,
        block_size: int = 32,
        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            vocab_size=NumericTokenizer.VOCAB_SIZE,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            pad_token_id=NumericTokenizer.PAD,
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError(
                    "Targets must have the same shape as inputs."
                )

            allowed = (
                targets.eq(-100)
                | targets.eq(NumericTokenizer.EOS)
                | (
                    (targets >= NumericTokenizer.DIGIT_OFFSET)
                    & (targets < NumericTokenizer.VOCAB_SIZE)
                )
            )

            if not torch.all(allowed):
                raise ValueError(
                    "Arithmetic targets may contain only digits, "
                    "EOS, or -100."
                )

        return super().forward(idx, targets)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 8,
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation for a single addition prompt.

        Only digit tokens and EOS are allowed in the generated response.
        EOS is forbidden until at least one digit has been generated.
        """

        if self.training:
            raise RuntimeError(
                "Call model.eval() before generation."
            )

        if prompt.ndim != 2 or prompt.shape[0] != 1:
            raise ValueError(
                "Generation currently supports batch size 1."
            )

        if prompt.dtype != torch.long:
            raise TypeError("Prompt must use torch.long.")

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")

        if prompt.shape[1] > self.block_size:
            raise ValueError("Prompt exceeds model context length.")

        # Validate the arithmetic prompt structure.
        NumericTokenizer.decode_input(prompt[0].tolist())

        output = prompt.clone()

        for generated_count in range(max_new_tokens):
            context = output[:, -self.block_size:]

            logits, _ = self(context)

            next_logits = logits[:, -1, :].clone()

            # Permit only digits and EOS.
            allowed = torch.zeros(
                self.vocab_size,
                dtype=torch.bool,
                device=output.device,
            )

            allowed[
                NumericTokenizer.DIGIT_OFFSET:
                NumericTokenizer.VOCAB_SIZE
            ] = True

            # Do not allow an empty answer.
            if generated_count > 0:
                allowed[NumericTokenizer.EOS] = True

            next_logits[:, ~allowed] = float("-inf")

            next_token = torch.argmax(
                next_logits,
                dim=-1,
                keepdim=True,
            )

            output = torch.cat(
                (output, next_token),
                dim=1,
            )

            if next_token.item() == NumericTokenizer.EOS:
                break

        return output