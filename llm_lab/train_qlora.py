import os
import random
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# --------- Config simple ---------
BASE_MODEL = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-v0.1")
SUBJECT = os.environ.get("MMLU_SUBJECT", "machine_learning")   # ex: "machine_learning"
OUT_DIR = os.environ.get("OUT_DIR", "./mmlu_qlora_out")

MAX_LEN = 512
SEED = 42

# --------- Helpers ---------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_split(ds_dict):
    # MMLU a parfois "auxiliary_train", parfois "train". On prend ce qui existe.
    for name in ["auxiliary_train", "train"]:
        if name in ds_dict:
            return name
    raise ValueError(f"Aucun split train trouvé. Splits disponibles: {list(ds_dict.keys())}")

def format_mmlu_example(ex):
    """
    ex typique MMLU:
      - question: str
      - choices: list[str] (4)
      - answer: int (0..3)
    """
    q = ex["question"].strip()
    choices = ex["choices"]
    # prompt
    prompt = (
        "Answer the multiple choice question by replying with A, B, C, or D only.\n\n"
        f"Question: {q}\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n"
        "Answer: "
    )
    # label (lettre)
    letter = "ABCD"[int(ex["answer"])]
    return prompt, letter

def tokenize_with_label_mask(tokenizer, prompt: str, answer: str):
    """
    On entraîne seulement sur la partie réponse (la lettre), en masquant le prompt.
    """
    full = prompt + answer
    tok = tokenizer(
        full,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )

    labels = tok["input_ids"].copy()

    # longueur du prompt en tokens (sans padding)
    prompt_tok = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )
    # compte tokens non-pad
    pad_id = tokenizer.pad_token_id
    prompt_len = 0
    for t in prompt_tok["input_ids"]:
        if t == pad_id:
            break
        prompt_len += 1

    # mask du prompt
    labels[:prompt_len] = [-100] * prompt_len
    tok["labels"] = labels
    return tok

def main():
    set_seed(SEED)

    # 1) Dataset
    ds = load_dataset("cais/mmlu", SUBJECT)
    train_split = pick_split(ds)
    train_ds = ds[train_split]

    # mini test (Kaggle): prends un sous-ensemble pour valider le pipeline
    # enlève ou augmente ensuite
    train_ds = train_ds.shuffle(seed=SEED).select(range(min(5000, len(train_ds))))

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) QLoRA (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # T4 => fp16
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    # 4) LoRA config (couvre Mistral/Llama-like)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # 5) Préprocess
    def map_fn(ex):
        prompt, ans = format_mmlu_example(ex)
        return tokenize_with_label_mask(tokenizer, prompt, ans)

    tokenized = train_ds.map(map_fn, remove_columns=train_ds.column_names)

    # 6) Training
    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        report_to="none",
        optim="paged_adamw_8bit",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    trainer.train()

    # 7) Sauvegarde LoRA adapters
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
