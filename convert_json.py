from datasets import load_dataset
from tqdm import tqdm
import time

ds = load_dataset("OpenSciLM/OpenScholar-DataStore-V3")

for split, dataset in ds.items():
    #files = dataset.to_json(f"my-dataset-{split}.jsonl")
    files = dataset.load_from_disk('$SCRATCH/OpenScholar/data/json')
    for i in tqdm (range (len(files))): #for i in files:
        time.sleep(0.1)
    
    print(f"Saved {len(files)} files to {files}")