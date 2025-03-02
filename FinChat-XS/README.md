# FinChat-XS
**FinChat-XS** is a lightweight financial domain language model designed to answer questions about finance, markets, investments, and economics in a conversational style.

[![Download Model](https://img.shields.io/badge/Download%20Model-Hugging%20Face-blue?style=for-the-badge&logo=HuggingFace)](https://huggingface.co/oopere/FinChat-XS)

## Model Overview. 
FinChat-XS is a fine-tuned version of [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), optimized for financial domain conversations using LoRA (Low-Rank Adaptation). With only 360M parameters, it offers a balance between performance and efficiency, making it accessible for deployment on consumer hardware.

The model combines professional financial knowledge with a conversational communication style, making it suitable for applications where users need expert financial information delivered in an approachable manner.

## How the Model was Created

FinChat-XS was developed through a focused fine-tuning process designed to enhance financial domain expertise while maintaining conversational abilities:

1. **Base model selection**: Started with SmolLM2-360M-Instruct, a lightweight instruction-tuned language model
2. **Dataset preparation**:
   - Filtered the sujet-ai/Sujet-Finance-Instruct-177k dataset to focus on QA and conversational QA examples
   - Applied length filtering to keep responses below 500 characters
   - Augmented short conversational QA examples to improve conciseness

3. **Fine-tuning approach**:
   - Applied LoRA (Low-Rank Adaptation) to efficiently fine-tune the model
   - Targeted key attention modules (q_proj, v_proj)
   - Used rank r=4 and alpha=16
   - Training configuration:
     - Batch size: 2 (effective batch size 16 with gradient accumulation)
     - Learning rate: 1.5e-4
     - BF16 precision

## Challenges & Future work. 
The primary challenge encountered during the development of FinChat-XS was the lack of high-quality conversational datasets specifically focused on personal finance. While the Sujet-Finance-Instruct-177k dataset provided valuable financial QA examples, there remains a notable gap in naturalistic, multi-turn conversations about personal financial scenarios.

To address this limitation, future work will focus on:

1. **Creating a specialized personal finance conversation dataset** that captures realistic interactions about budgeting, investing, retirement planning, debt management, and other everyday financial concerns.

2. **Developing a synthetic data generation tool** that leverages various LLMs to create diverse, high-quality conversational data in the personal finance domain. This tool will help produce training examples that better reflect how real users interact when discussing their financial questions and concerns.

This work aims to significantly improve the ability of financial assistant models to engage in helpful, informative conversations about personal finance while maintaining accessibility through smaller model sizes.

## Why Use This Model?

FinChat-XS offers several advantages for specific use cases:

- **Efficient deployment**: At only 362MB, it can run on devices with limited resources. 
- **Financial domain knowledge**: Fine-tuned specifically on financial QA data
- **Balanced communication style**: Combines professional financial knowledge with conversational delivery
- **Low deployment cost**: Requires significantly less computational resources than larger models
- **Customizable**: The LoRA adapter can be mixed with other adapters or further fine-tuned

Ideal for:
- Embedded financial assistants in mobile apps
- Personal financial planning tools
- Educational applications about finance and investing
- Customer service automation for financial institutions
- Quick deployment scenarios where larger models aren't practical

## How to Use the Model

### Basic Usage with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "oopere/FinChat-XS"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Create a conversation
messages = [
    {"role": "user", "content": "What's the difference between stocks and bonds?"}
]

# Format the prompt using the chat template
prompt = tokenizer.apply_chat_template(messages, tokenize=False)

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate a response
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.2
)

# Decode and print the response
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

### Optimized Inference with 8-bit Quantization

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Configure 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "oopere/FinChat-XS", 
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("oopere/FinChat-XS")

# Continue with the same usage pattern as above
```

### Using with LoRA Adapter Only

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")

# Load LoRA adapter
peft_model = PeftModel.from_pretrained(base_model, "oopere/qa-adapterFinChat-XS")

# Continue with the same usage pattern as above
```

## Limitations & Considerations

While FinChat-XS performs well in many financial conversation scenarios, users should be aware of these limitations:

1. **Knowledge limitations**: The model's knowledge is limited to its training data and has a knowledge cutoff date from the base model (SmolLM2).

2. **Size trade-offs**: As a 360M parameter model, it has less capacity than larger models (7B+) and may provide less nuanced or detailed responses on complex topics.

3. **Financial advice disclaimer**: The model is not a certified financial advisor and should not be used for making investment decisions. Its responses should be considered educational, not professional financial advice.

4. **Domain boundaries**: While focused on finance, the model may struggle with highly specialized financial topics or recent developments not covered in its training data.

5. **Hallucination potential**: Like all language models, FinChat-XS may occasionally generate plausible-sounding but incorrect information, especially when asked about specific numerical data or complex financial details.

6. **Style variations**: The model balances formal financial knowledge with a conversational style, which may not be appropriate for all professional contexts.

7. **Regulatory compliance**: This model has not been specifically audited for compliance with financial regulations in various jurisdictions.

## Citation

If you use FinChat-XS in your research or applications, please consider citing it as:

```bibtex
@misc{oopere2025finchatxs,
  author = {Martra, P.},
  title = {FinChat-XS: A Lightweight Financial Domain Chat Language Model},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/oopere/FinChat-XS}}
}
```
