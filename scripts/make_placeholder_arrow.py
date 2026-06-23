"""
jsonl -> Arrow (placeholder entropies) for entropy model training.

Entropy model training (add_patches=false) never reads the entropies field,
so we fill it with empty float16 lists just to satisfy the Arrow schema.

Output path follows arrow_iterator.py convention:
  {preprocess_dir}/{dataset}/{entropy_model_name}/{chunk_file}.shard_0.arrow
  {preprocess_dir}/{dataset}/{entropy_model_name}/{chunk_file}.shard_0.arrow.complete
"""
import os
import re
import sys

import jsonlines
import numpy as np
import pyarrow as pa

ENTROPY_MODEL_NAME = "transformer_100m"

sample_id_field = pa.field("sample_id", pa.string(), nullable=False)
text_field = pa.field("text", pa.string(), nullable=False)
entropy_field = pa.field("entropies", pa.list_(pa.float16()), nullable=False)
SCHEMA = pa.schema([sample_id_field, text_field, entropy_field])

PLACEHOLDER_ENTROPIES = np.array([], dtype=np.float16)


def get_text(doc: dict) -> str:
    if "text" in doc:
        return doc["text"]
    elif "content" in doc:
        return doc["content"]
    raise ValueError(f"No text field in: {list(doc.keys())}")


def convert(jsonl_path: str, preprocess_dir: str, batch_size: int = 2000):
    basename = os.path.basename(jsonl_path)
    parts = re.match(r"(.+)\.chunk\.[0-9]+\.jsonl", basename)
    assert parts, f"Filename does not match *.chunk.*.jsonl pattern: {basename}"
    dataset = parts.group(1)

    data_dir = os.path.join(preprocess_dir, dataset, ENTROPY_MODEL_NAME)
    os.makedirs(data_dir, exist_ok=True)

    output_file = os.path.join(data_dir, f"{basename}.shard_0.arrow")
    complete_file = f"{output_file}.complete"

    if os.path.exists(complete_file):
        print(f"Already done: {output_file}")
        return

    print(f"Reading: {jsonl_path}")
    print(f"Writing: {output_file}")

    id_buf, text_buf, entropy_buf = [], [], []
    total = 0

    with open(output_file, "wb") as sink:
        with pa.ipc.new_file(sink, SCHEMA) as writer:
            with jsonlines.open(jsonl_path) as reader:
                for i, doc in enumerate(reader):
                    id_buf.append(str(i))
                    text_buf.append(get_text(doc))
                    entropy_buf.append(PLACEHOLDER_ENTROPIES)
                    total += 1

                    if len(id_buf) == batch_size:
                        writer.write(
                            pa.record_batch(
                                {"sample_id": id_buf, "text": text_buf, "entropies": entropy_buf},
                                schema=SCHEMA,
                            )
                        )
                        id_buf, text_buf, entropy_buf = [], [], []

                if id_buf:
                    writer.write(
                        pa.record_batch(
                            {"sample_id": id_buf, "text": text_buf, "entropies": entropy_buf},
                            schema=SCHEMA,
                        )
                    )

    open(complete_file, "w").close()
    print(f"Done. {total} records written.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <jsonl_path> <preprocess_dir>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
