import os
import sys
import json
import io
import zstandard as zstd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Install via: pip install tqdm

def decompress_file(args):
    zst_file_path, output_jsonl_path = args
    dctx = zstd.ZstdDecompressor()

    try:
        with open(zst_file_path, 'rb') as compressed:
            with dctx.stream_reader(compressed) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')

                with open(output_jsonl_path, 'w', encoding='utf-8') as out_file:
                    for line in text_stream:
                        try:
                            data = json.loads(line)
                            out_file.write(json.dumps(data) + '\n')
                        except json.JSONDecodeError:
                            continue
        return f"✔ {os.path.basename(zst_file_path)}"
    except Exception as e:
        return f"✖ {os.path.basename(zst_file_path)} — {e}"

def decompress_folder_zst_to_jsonl_parallel(input_folder, output_folder, num_workers=None):
    os.makedirs(output_folder, exist_ok=True)

    zst_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith('.zst')
    ]

    tasks = [
        (zst_file, os.path.join(output_folder, os.path.basename(zst_file).replace('.zst', '.jsonl')))
        for zst_file in zst_files
    ]

    print(f"🔄 Processing {len(tasks)} files using {num_workers or cpu_count()} workers...\n")

    with Pool(processes=num_workers or cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(decompress_file, tasks), total=len(tasks), desc="Decompressing"):
            tqdm.write(result)  # Print results without breaking the progress bar

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python decompress_zst_to_jsonl.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist or is not a directory.")
        sys.exit(1)

    decompress_folder_zst_to_jsonl_parallel(input_folder, output_folder)
    print(f"\n✅ All files processed. Output saved to: {output_folder}")
