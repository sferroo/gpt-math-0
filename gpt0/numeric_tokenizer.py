from __future__ import annotations

from typing import Sequence


class NumericTokenizer:
    """
    Tokenizer dedicated to decimal arithmetic.

    Vocabulary:
        PAD: 0
        BOS: 1
        EOS: 2
        ADD: 3
        EQ:  4
        0-9: 5-14
    """

    PAD = 0
    BOS = 1
    EOS = 2
    ADD = 3
    EQ = 4

    DIGIT_OFFSET = 5
    VOCAB_SIZE = 15

    TOKEN_TO_ID = {
        "<PAD>": 0,
        "<BOS>": 1,
        "<EOS>": 2,
        "<ADD>": 3,
        "<EQ>": 4,
        **{
            str(digit): 5 + digit
            for digit in range(10)
        },
    }

    ID_TO_TOKEN = {
        token_id: token
        for token, token_id in TOKEN_TO_ID.items()
    }

    @classmethod
    def encode_digits(cls, number: int) -> list[int]:
        if not isinstance(number, int) or isinstance(number, bool):
            raise TypeError("Number must be an integer.")

        if number < 0:
            raise ValueError("Negative numbers are not supported.")

        return [
            cls.DIGIT_OFFSET + int(digit)
            for digit in str(number)
        ]

    @classmethod
    def encode_input(cls, a: int, b: int) -> list[int]:
        return (
            [cls.BOS]
            + cls.encode_digits(a)
            + [cls.ADD]
            + cls.encode_digits(b)
            + [cls.EQ]
        )

    @classmethod
    def encode_target(cls, result: int) -> list[int]:
        return cls.encode_digits(result) + [cls.EOS]

    @classmethod
    def encode_sequence(cls, a: int, b: int) -> list[int]:
        result = a + b
        return (
            cls.encode_input(a, b)
            + cls.encode_target(result)
        )

    @classmethod
    def decode_digits(cls, token_ids: Sequence[int]) -> str:
        digits = []

        for token_id in token_ids:
            if token_id == cls.EOS:
                break

            if not cls.DIGIT_OFFSET <= token_id < cls.VOCAB_SIZE:
                raise ValueError(
                    f"Invalid digit token ID: {token_id}"
                )

            digits.append(str(token_id - cls.DIGIT_OFFSET))

        if not digits:
            raise ValueError("No digits found in token sequence.")

        return "".join(digits)

    @classmethod
    def decode(cls, token_ids: Sequence[int]) -> int:
        return int(cls.decode_digits(token_ids))

    @classmethod
    def validate_encoding(
        cls,
        a: int,
        b: int,
    ) -> bool:
        input_ids = cls.encode_input(a, b)
        target_ids = cls.encode_target(a + b)

        decoded_input = cls.decode_input(input_ids)
        decoded_target = cls.decode(target_ids)

        return (
            decoded_input == (a, b)
            and decoded_target == a + b
        )

    @classmethod
    def decode_input(
        cls,
        token_ids: Sequence[int],
    ) -> tuple[int, int]:
        if len(token_ids) < 5:
            raise ValueError("Input sequence is too short.")

        if token_ids[0] != cls.BOS:
            raise ValueError("Missing BOS token.")

        if token_ids[-1] != cls.EQ:
            raise ValueError("Missing EQ token.")

        try:
            add_position = token_ids.index(cls.ADD)
        except ValueError as exc:
            raise ValueError("Missing ADD token.") from exc

        if add_position <= 1 or add_position >= len(token_ids) - 2:
            raise ValueError("Invalid operand positions.")

        if token_ids.count(cls.ADD) != 1:
            raise ValueError("Expected exactly one ADD token.")

        if any(
            token_id < cls.DIGIT_OFFSET
            or token_id >= cls.VOCAB_SIZE
            for token_id in token_ids[1:add_position]
        ):
            raise ValueError("Invalid digit in first operand.")

        if any(
            token_id < cls.DIGIT_OFFSET
            or token_id >= cls.VOCAB_SIZE
            for token_id in token_ids[add_position + 1:-1]
        ):
            raise ValueError("Invalid digit in second operand.")

        a = cls.decode_digits(token_ids[1:add_position])
        b = cls.decode_digits(token_ids[add_position + 1:-1])

        return int(a), int(b)