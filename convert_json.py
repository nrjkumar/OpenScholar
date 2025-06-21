# from datasets import load_dataset
# from tqdm import tqdm
# import time

# ds = load_dataset("OpenSciLM/OpenScholar-DataStore-V3")

# for split, dataset in ds.items():
#     #files = dataset.to_json(f"my-dataset-{split}.jsonl")
#     files = dataset.load_from_disk('$SCRATCH/OpenScholar/data/json')
#     for i in tqdm (range (len(files))): #for i in files:
#         time.sleep(0.1)
    
#     print(f"Saved {len(files)} files to {files}")
    
import json , argparse

def json_to_jsonl(json_file_path, jsonl_file_path): #input_file, output_file):
#     with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
#         for line in f_in:
#             data = json.loads(line)
#             for item in data:
#                 f_out.write(json.dumps(item) + '\n')

    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        with jsonlines.open(jsonl_file_path, mode='w') as writer:
            if isinstance(data, list):
                for item in data:
                    writer.write(item)
            elif isinstance(data, dict):
                writer.write(data)
            else:
                print("Error: Input JSON must be a list or a dictionary.")

    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

                
def main():
    parser = argparse.ArgumentParser(description='Convert JSON file to JSONL file')
    parser.add_argument('-i', '--input', help='Input JSON file', required=True)
    parser.add_argument('-o', '--output', help='Output JSONL file', required=True)
    args = parser.parse_args()

    json_to_jsonl(args.input, args.output)

if __name__ == '__main__':
    main()