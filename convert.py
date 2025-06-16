import os
import zstandard as zstd
import io
import json

def decompress_folder_zst_to_jsonl(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    dctx = zstd.ZstdDecompressor()

    for filename in os.listdir(input_folder):
        if filename.endswith('.zst'):
            zst_file_path = os.path.join(input_folder, filename)

            # Each output .jsonl file has same base name as .zst file
            output_jsonl_path = os.path.join(output_folder, filename.replace('.zst', '.jsonl'))

            with open(zst_file_path, 'rb') as compressed:
                with dctx.stream_reader(compressed) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')

                    with open(output_jsonl_path, 'w', encoding='utf-8') as out_file:
                        for line in text_stream:
                            try:
                                data = json.loads(line)
                                out_file.write(json.dumps(data) + '\n')  # One JSON object per line
                            except json.JSONDecodeError:
                                continue  # Skip malformed JSON lines
