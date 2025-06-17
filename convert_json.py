from datasets import load_dataset

ds = load_dataset("OpenSciLM/OpenScholar-DataStore-V3")



for split, dataset in ds.items():
    files = dataset.to_json(f"my-dataset-{split}.jsonl")
    files.save_to_disk('$SCRATCH/OpenScholar/data/json')
    
    print(f"Saved {len(files)} files to {files}")