"""The data types of the TDG schema (`dtype`) and what the cost models need to know about them."""

BYTES_PER_ELEMENT = {"int8": 1, "uint8": 1, "int16": 2, "int32": 4, "fp16": 2, "fp32": 4}


def bytes_per_element(dtype: str) -> int:
    return BYTES_PER_ELEMENT[dtype]


def bit_width(dtype: str) -> int:
    return 8 * BYTES_PER_ELEMENT[dtype]


def is_floating_point(dtype: str) -> bool:
    return dtype.startswith("fp")


def widest(dtypes: list[str]) -> str:
    """The dtype with the most bits; floating point wins a tie. Used as the 'compute dtype'."""
    return max(dtypes, key=lambda dtype: (bit_width(dtype), is_floating_point(dtype)))
