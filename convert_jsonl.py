
import json
import multiprocessing
from tqdm import tqdm


def parse_json_line(line):
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError:
        return None


def process_jsonl_parallel(input_path, output_path, batch_size=10000):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

        lines = []
        total_written = 0
        with tqdm(desc="Processing", unit=" lines") as pbar:
            for line in infile:
                lines.append(line)
                if len(lines) >= batch_size:
                    results = pool.map(parse_json_line, lines)
                    for obj in results:
                        if obj:
                            outfile.write(json.dumps(obj, ensure_ascii=False) + '\n')
                            total_written += 1
                    pbar.update(len(lines))
                    lines = []

            # Final flush
            if lines:
                results = pool.map(parse_json_line, lines)
                for obj in results:
                    if obj:
                        outfile.write(json.dumps(obj, ensure_ascii=False) + '\n')
                        total_written += 1
                pbar.update(len(lines))

            pool.close()
            pool.join()
            print(f"\n✅ Done. Total lines written: {total_written}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stream large JSON array to JSONL format")
    parser.add_argument("input", help="Path to large JSON array file (e.g., huge.json)")
    parser.add_argument("output", help="Path to output JSONL file (e.g., huge.jsonl)")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for multiprocessing")
    args = parser.parse_args()

    process_jsonl_parallel(args.input, args.output, batch_size=args.batch_size)