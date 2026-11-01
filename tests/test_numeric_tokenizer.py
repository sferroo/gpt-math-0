import pytest

from gpt0.numeric_tokenizer import NumericTokenizer


def test_vocabulary_size():
    assert NumericTokenizer.VOCAB_SIZE == 15
    assert len(NumericTokenizer.TOKEN_TO_ID) == 15


def test_digit_encoding():
    assert NumericTokenizer.encode_digits(123) == [6, 7, 8]


def test_input_encoding():
    assert NumericTokenizer.encode_input(78, 45) == [
        1, 12, 13, 3, 9, 10, 4
    ]


def test_target_encoding():
    assert NumericTokenizer.encode_target(123) == [
        6, 7, 8, 2
    ]


def test_full_sequence():
    assert NumericTokenizer.encode_sequence(78, 45) == [
        1, 12, 13, 3, 9, 10, 4, 6, 7, 8, 2
    ]


def test_decode_digits():
    assert NumericTokenizer.decode_digits([6, 7, 8, 2]) == "123"


def test_decode_result():
    assert NumericTokenizer.decode([6, 7, 8, 2]) == 123


def test_decode_input():
    tokens = NumericTokenizer.encode_input(78, 45)

    assert NumericTokenizer.decode_input(tokens) == (78, 45)


@pytest.mark.parametrize(
    "a,b",
    [
        (0, 0),
        (1, 2),
        (78, 45),
        (99, 99),
        (1234, 5678),
        (9999, 9999),
    ],
)
def test_roundtrip(a, b):
    assert NumericTokenizer.validate_encoding(a, b)


def test_negative_number_rejected():
    with pytest.raises(ValueError):
        NumericTokenizer.encode_digits(-5)


def test_invalid_digit_token_rejected():
    with pytest.raises(ValueError):
        NumericTokenizer.decode_digits([15, 2])


def test_missing_bos_rejected():
    with pytest.raises(ValueError):
        NumericTokenizer.decode_input([12, 13, 3, 9, 10, 4])


def test_missing_eq_rejected():
    with pytest.raises(ValueError):
        NumericTokenizer.decode_input([1, 12, 13, 3, 9, 10])


def test_empty_target_rejected():
    with pytest.raises(ValueError):
        NumericTokenizer.decode_digits([2])