import tiktoken

# TODO: try to adapt to other self-deployed models
TOKENIZERS = {}


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global TOKENIZERS
    if model_name not in TOKENIZERS:
        TOKENIZERS[model_name] = tiktoken.encoding_for_model(model_name)
    tokens = TOKENIZERS[model_name].encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global TOKENIZERS
    if model_name not in TOKENIZERS:
        TOKENIZERS[model_name] = tiktoken.encoding_for_model(model_name)
    content = TOKENIZERS[model_name].decode(tokens)
    return content


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(range(0, len(tokens), max_token_size - overlap_token_size)):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results
