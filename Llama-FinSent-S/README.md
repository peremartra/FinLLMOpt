# Llama-FinSent-S

**Llama-FinSent-S** is a fine-tuned and pruned version of LLaMA-3.2-1B, designed for financial sentiment analysis. It balances efficiency and performance, making it a suitable choice for financial applications that require accurate sentiment classification with reduced computational costs.

[![Download Model](https://img.shields.io/badge/Download%20Model-Hugging%20Face-blue?style=for-the-badge&logo=HuggingFace)](https://huggingface.co/oopere/Llama-FinSent-S)

## Model Overview

Llama-FinSent-S is based on `oopere/pruned40-llama-1b`, a pruned version of LLaMA-3.2-1B. The pruning process reduces the number of neurons in the MLP layers by **40%**, leading to:
- **Lower power consumption** and improved efficiency.
- **Retained competitive performance** in reasoning and instruction-following tasks.

Additionally, the **MLP expansion ratio** has been reduced from **300% to 140%**, an optimal trade-off identified in the paper *Exploring GLU Expansion Ratios: Structured Pruning in Llama-3.2 Models*.

## Training Process

The model was developed through a two-step process:

1. **Pruning**: The base LLaMA-3.2-1B model underwent structured pruning, reducing its MLP neurons by 40% to decrease computational requirements while preserving key capabilities.
2. **Fine-Tuning with LoRA**: The pruned model was fine-tuned using **LoRA (Low-Rank Adaptation)** on the **FinGPT/fingpt-sentiment-train** dataset. After training, the LoRA adapter was merged into the base model, resulting in a compact yet efficient model.

The **fine-tuning process** required just **40 minutes** on an A100 GPU while maintaining high-quality sentiment classification performance.

## Why Use This Model?

- **Efficiency**: Pruned architecture reduces computational costs and memory footprint.
- **Performance Gains**: Retains or improves performance in key areas such as **instruction-following (IFEVAL)** and **multi-step reasoning (MUSR)**.
- **Financial Domain Optimization**: Specifically fine-tuned for financial sentiment classification.
- **Flexible Sentiment Classification**: Supports **seven-category (fine-grained)** and **three-category (coarse)** sentiment labeling.

## How to Use the Model

### Installation

Ensure you have the required dependencies installed:

```bash
pip install transformers torch
```

### Load the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Model and tokenizer
model_name = "oopere/Llama-FinSent-S"  
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### Perform Sentiment Classification

```python
def generate_response(prompt, model, tokenizer):
    """Generates sentiment classification response."""
    full_prompt = (
        "Instruction: What is the sentiment of this news? "
        "Please choose an answer from {strong negative/moderately negative/mildly negative/neutral/"
        "mildly positive/moderately positive/strong positive}."
        "\n" + "News: " + prompt + "\n" + "Answer:"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=15,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        temperature=0.001,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_response.split("Answer:")[-1].strip()

# Example usage
news_text = "Ahlstrom Corporation STOCK EXCHANGE ANNOUNCEMENT 7.2.2007 at 10.30 A total of 56,955 new shares of A..."
sentiment = generate_response(news_text, model, tokenizer)
print("Predicted Sentiment:", sentiment)
```

### Alternative: Three-Class Sentiment Classification

```python
full_prompt = (
    "Instruction: What is the sentiment of this news? "
    "Please choose an answer from {negative/neutral/positive}."
    "\n" + "News: " + prompt + "\n" + "Answer:"
)
```

## Limitations & Considerations

- **Domain-Specific**: Optimized for financial texts; performance may degrade on general sentiment tasks.
- **Potential Biases**: Training data may contain inherent biases affecting predictions.
- **Hardware Requirements**: While pruned, running inference on a CPU might be slower than on a GPU.

## Citation

If you use this model in your research or applications, please cite:

```bibtex
@misc{Llama-FinSent-S,
  title={Llama-FinSent-S: A Pruned LLaMA-3.2 Model for Financial Sentiment Analysis},
  author={Pere Martra},
  year={2025},
  url={https://huggingface.co/your-hf-username/Llama-FinSent-S}
}

@misc{Martra2024,
  author={Martra, P.},
  title={Exploring GLU Expansion Ratios: Structured Pruning in Llama-3.2 Models},
  year={2024},
  url={https://doi.org/10.31219/osf.io/qgxea}
}
```

## License

This model follows the licensing terms of LLaMA models. While this repository is under **Apache-2.0**, models derived from LLaMA may have additional licensing restrictions. Please review the respective model cards for details.



