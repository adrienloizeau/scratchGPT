import json
from datasets import load_dataset
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters import LanguageFilter
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import LocalPipelineExecutor
from tqdm import tqdm

def stack_dataset(ds):
    split = ["train", "test", "validation"]
    cleaned_data = []
    for s in split:  
        for i in tqdm(range(len(ds[s]["title"]))):  
            cleaned_data.append({
                "text": f"{ds[s]['title'][i]}\n{ds[s]['body'][i]}",
                "source": "law_stack_exchange",
                "score": ds[s]["Score"][i]
            })
    return cleaned_data

def save_dataset(ds: list, path: str):
    """Save dataset in proper JSONL format"""
    with open(path, "w", encoding="utf-8") as f:
        for item in ds:
            json.dump(item, f)
            f.write("\n")


class ScoreFilter(PipelineStep):
    def __init__(self, min_score: int):
        self.min_score = min_score

    def run(self, data, rank: int = 0, world_size: int = 1):
        for document in data:
            if "score" in document.metadata:
                if int(document.metadata["score"]) > self.min_score:
                    yield document
            else:   
                yield document

# Law Stack Exchange dataset
save_dataset_path = "data/law_stack_exchange.jsonl"
signatures_folder = "data/signatures" 
output_path = "data/cleaned_documents.jsonl"

ds = load_dataset("jonathanli/law-stack-exchange")
stacked_ds = stack_dataset(ds)

# Fineweb dataset
save_dataset_path = "data/fineweb.jsonl"
output_path = "cleaned_dataset.jsonl"
ds = load_dataset("HuggingFaceFW/fineweb", "CC-MAIN-2013-48", split="train", streaming=True)
stacked_ds = []
for i, example in enumerate(ds):
    if i >= 10000:  # Stop after 1000 examples (adjust based on your needs)
        break
    stacked_ds.append(example)

save_dataset(stacked_ds, save_dataset_path)


pipeline = [
    JsonlReader(data_folder="data"),
    ScoreFilter(min_score=1), # Filter out low-score documents
    LanguageFilter(languages="en"),
    JsonlWriter(output_folder="data", output_filename=output_path, compression=None),
]
executor = LocalPipelineExecutor(pipeline=pipeline, tasks=1, workers=1)
executor.run()

print("Running pipeline...")
executor.run()
print(f"Pipeline complete. Check {output_path}")

# Verify output
import os
print(f"Output file exists: {os.path.exists(output_path)}")
print(f"File size: {os.path.getsize(output_path) if os.path.exists(output_path) else 0} bytes")