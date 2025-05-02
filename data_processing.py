import json
import datasets
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters import LanguageFilter, GopherQualityFilter, GopherRepetitionFilter
from datatrove.pipeline.dedup import MinhashDedupSignature, MinhashDedupFilter
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

class MergeFields(PipelineStep):
    def run(self, data, rank: int = 0, world_size: int = 1):
        for document in data:
            document["text"] = f"{document['title']}\n{document['body']}"
            yield document

class ScoreFilter(PipelineStep):
    def __init__(self, min_score: int):
        self.min_score = min_score

    def run(self, data, rank: int = 0, world_size: int = 1):
        for document in data:
            if int(document.metadata["score"]) > self.min_score:
                yield document

# Path configuration
save_dataset_path = "data/law_stack_exchange.jsonl"
signatures_folder = "data/signatures" 
output_path = "data/cleaned_documents.jsonl"

# Uncomment to process and save raw data
# ds = datasets.load_dataset("jonathanli/law-stack-exchange")
# stacked_ds = stack_dataset(ds)
# save_dataset(stacked_ds, save_dataset_path)

pipeline = [
    JsonlReader(data_folder="data"),
    ScoreFilter(min_score=1), # Minimum score filter
    # GopherQualityFilter(),  # no gpu 
    # GopherRepetitionFilter(),  # no gpu 
    LanguageFilter(languages="en"),
    JsonlWriter(output_folder="data", output_filename="cleaned_documents.jsonl", compression=None),
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