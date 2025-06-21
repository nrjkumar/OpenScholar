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
    
import json

def json_to_jsonl(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            for item in data:
                f_out.write(json.dumps(item) + '\n')