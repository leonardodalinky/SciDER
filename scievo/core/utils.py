import toon


def wrap_dict_to_toon(d: dict) -> str:
    s = toon.encode(d)
    if s == "null":
        raise ValueError("Failed to encode dict to TOON")
    return f"```toon\n{s}\n```"
