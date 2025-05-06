# ScratchGPT

ScratchGPT is a project to build a transformer-based language model from scratch, incorporating key steps like **pretraining**, **instruction fine-tuning**, and **data filtering** to ensure high-quality datasets. Inspired by [Andrej Karpathy's NanoGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY), it replicates modern language model workflows.

---

## Training Strategies

### Pretraining
- **Datasets**: FineWeb and small Law Stack Exchange to introduce some instruct.
- **Pipeline**:
  - **Data Loading**: Extract and prepare raw data.
  - **Tokenization**: Encode sequences with `CharTokenizer`.
  - **Training**: Cross-Entropy loss to predict the next token.

### Instruction Fine-Tuning
- **Datasets**: Dolly 15k, Alpaca.
- **Pipeline**:
  - Add `|||` token to separate instructions from responses.
  - Focus loss computation on the response to avoid instruction copying.

---

## Dataset Processing

### Filtering done with `datatrove`:
- **Language**: Keep only English documents using `LanguageFilter`.
- **Quality**: Remove irrelevant or noisy data (e.g., off-topic Law Stack Exchange discussions).
- (wip) fast classifier

---

## Configurations

- **LargeConfig**: Optimized for A100 GPUs, delivers reasonable results.
- **SmallConfig**: Runs locally, capable of generating simple words.

---

## Future Improvements
- **Data Filtering**: Use the filtering applied for Fineweb on the datatrove lib.
- **Direct Preference Optimization (DPO)**:  align a model without RLHF
- **Chinchilla Law**: Balance model size and dataset scale for efficiency.
- **Evaluation**: Add metrics to assess performance.
