import toon


def wrap_text_with_block(s: str, block_name: str) -> str:
    return f"```{block_name}\n{s}\n```"


def wrap_dict_to_toon(d: dict) -> str:
    s = toon.encode(d)
    if s == "null":
        raise ValueError("Failed to encode dict to TOON")
    return wrap_text_with_block(s, "toon")
