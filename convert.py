import os
import sys
import zstandard as zstd
import io
import json

def decompress_folder_zst_to_jsonl(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    dctx = zstd.ZstdDecompressor()

    for filename in os.listdir(input_folder):
        if filename.endswith('.zst'):
            zst_file_path = os.path.join(input_folder, filename)
            output_jsonl_path = os.path.join(output_folder, filename.replace('.zst', '.jsonl'))

            with open(zst_file_path, 'rb') as compressed:
                with dctx.stream_reader(compressed) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')

                    with open(output_jsonl_path, 'w', encoding='utf-8') as out_file:
                        for line in text_stream:
                            try:
                                data = json.loads(line)
                                out_file.write(json.dumps(data) + '\n')
                            except json.JSONDecodeError:
                                continue  # Skip malformed lines

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python decompress_zst_to_jsonl.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist or is not a directory.")
        sys.exit(1)

    decompress_folder_zst_to_jsonl(input_folder, output_folder)
    print(f"Decompression complete. Files saved in: {output_folder}")
