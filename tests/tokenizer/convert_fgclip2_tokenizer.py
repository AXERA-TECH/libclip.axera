#!/usr/bin/env python3
"""Convert the FG-CLIP2 Hugging Face tokenizer.json to libclip's tokenizer format.

FG-CLIP2 uses a Gemma-style BPE tokenizer.  libclip's SentencePiece reader can
consume the same vocabulary/merge table when the SentencePiece metaspace marker
(``U+2581``) is represented as a normal space and the BPE merge rank is encoded
as the piece score.
"""

import argparse
import base64
import json
import re


MAGIC_NUMBER = 430
SENTENCEPIECE = 0

NORMAL = 1
UNKNOWN = 2
CONTROL = 3
UNUSED = 5
BYTE = 6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer_json",
        default="fg-clip2-base/tokenizer.json",
        help="Path to fg-clip2-base/tokenizer.json",
    )
    parser.add_argument(
        "--dst",
        default="fgclip2_tokenizer.txt",
        help="Output tokenizer file consumed by libclip",
    )
    return parser.parse_args()


def load_bpe(tokenizer_json):
    with open(tokenizer_json, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    model = data.get("model", {})
    if model.get("type") != "BPE":
        raise ValueError(f"Expected a BPE tokenizer, got {model.get('type')!r}")

    vocab = model["vocab"]
    merges = model["merges"]
    if not isinstance(vocab, dict):
        raise ValueError("Expected tokenizer model.vocab to be a token -> id mapping")

    merge_rank = {}
    for rank, pair in enumerate(merges):
        if len(pair) != 2:
            raise ValueError(f"Invalid merge entry: {pair!r}")
        merged = pair[0] + pair[1]
        merge_rank.setdefault(merged, rank)
    return data, vocab, merge_rank


def special_ids(data):
    ids = {
        token["id"]
        for token in data.get("added_tokens", [])
        if token.get("special") and token.get("content") in {"<pad>", "<eos>", "<bos>", "<unk>"}
    }
    for token_id in (0, 1, 2, 3):
        if token_id not in ids:
            ids.add(token_id)
    return sorted(ids)


def write_tokenizer(dst, vocab, merge_rank, specials):
    id_to_token = {token_id: token for token, token_id in vocab.items()}
    if sorted(id_to_token) != list(range(len(id_to_token))):
        raise ValueError("Tokenizer vocabulary IDs must be contiguous from zero")

    byte_piece = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
    special_set = set(specials)

    with open(dst, "w", encoding="utf-8") as fp:
        # header: magic + tokenizer type
        fp.write(f"{MAGIC_NUMBER} {SENTENCEPIECE}\n")
        # header: special token count, stop token count, prefix token count
        fp.write(f"{len(specials)} 1 0\n")
        fp.write(" ".join(str(token_id) for token_id in specials))
        fp.write(" 1\n")
        fp.write(f"{len(id_to_token)}\n")

        for token_id in range(len(id_to_token)):
            token = id_to_token[token_id]
            piece = token.replace("\u2581", " ")

            if token_id in special_set:
                piece_type = CONTROL
            elif token == "<unk>":
                piece_type = UNKNOWN
            elif byte_piece.match(token):
                piece_type = BYTE
            else:
                piece_type = NORMAL

            # SentencePiece BPE picks the pair with the highest score.  Negative
            # merge rank therefore reproduces tokenizer.json's merge order.
            score = -float(merge_rank[token]) if token in merge_rank else 0.0
            encoded = base64.b64encode(piece.encode("utf-8")).decode("ascii")
            fp.write(f"{encoded} {score} {piece_type}\n")


def main():
    args = parse_args()
    data, vocab, merge_rank = load_bpe(args.tokenizer_json)
    specials = special_ids(data)
    write_tokenizer(args.dst, vocab, merge_rank, specials)
    print(f"Wrote {len(vocab)} tokens to {args.dst}")
    print(f"Special token ids: {specials}")
    print(f"BPE merges: {len(merge_rank)}")


if __name__ == "__main__":
    main()
