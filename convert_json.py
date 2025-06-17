from datasets import load_dataset
from tqdm import tqdm

ds = load_dataset("OpenSciLM/OpenScholar-DataStore-V3")

for split, dataset in ds.items():
    files = dataset.to_json(f"my-dataset-{split}.jsonl")
    files.save_to_disk('$SCRATCH/OpenScholar/data/json')
    for i in files:
        tqdm.write(i)
    
    print(f"Saved {len(files)} files to {files}")