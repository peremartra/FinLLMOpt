# 🏦 RetailBanking-Conversations Dataset (FinLLMOpt)

This dataset contains 320 realistic synthetic conversations between banking customers and financial advisors. It is part of the [FinLLMOpt](https://github.com/peremartra/FinLLMOpt) project and was built to support the development and evaluation of optimized Instruct LLMs for the financial sector.

---

## Contents
- `retail_banking_dataset.json`: The dataset in json format. 
- `README.md`: This file.
- Dataset hosted on Hugging Face: [RetailBanking-Conversations](https://huggingface.co/datasets/oopere/RetailBanking-Conversations)

---

## Overview

- **Conversations**: 320
- **Profiles**: 160 unique financial personas
- **Topics**: 10 main financial categories, including:
  - Cards
  - Bank Accounts
  - Mortgages
  - Investment Funds
  - Pension Plans
  - Insurance
  - Customer Rewards
  - Digital Banking
  - Savings & Deposits
  - Personal Loans

Each conversation simulates a customer interacting with a financial advisor to achieve a goal, ask questions, or clarify doubts. All data is fully synthetic and contains no sensitive or real-world information.

---

## How it was generated: WizardSData 🧙‍♂️

This dataset was created using [WizardSData](https://github.com/peremartra/WizardSData), an open-source Python library developed to simplify the generation of structured datasets using LLMs.

### Highlights of the generation process:
- **Two-role conversation simulation** (client ↔ advisor)
- **Role-specific prompt templates** using Jinja2
- **OpenAI GPT-4o** as the generation model
- **Controlled randomness** with different temperatures per role
- **Rich configuration via JSON + CLI/Python**

---

## Example Profile (used in generation)

```json
{
  "name": "Laura",
  "age": 42,
  "country": "Spain",
  "profession": "teacher",
  "financial_knowledge": "basic",
  "financial_products": ["savings account"],
  "financial_goal": "buy a second home",
  "investment_horizon": "medium term",
  "risk_tolerance": "low",
  "conversation_style": "formal",
  "marital_status": "married",
  "residence_area": "urban",
  "topic": "Mortgages"
}
```

---

## Prompt Templates
### `financial_client_01.j2`
Used to instruct the client role in each conversation.

Key behaviors:
- Speak naturally and reveal information gradually.
- Start with a short greeting.
- Follow a goal-oriented flow while remaining casual or formal depending on the persona.

Excerpt:
> You are a 42-year-old married client living in a urban area of Spain. You work as a teacher and have basic financial knowledge...

### `financial_advisor_01.j2`
Used for the advisor role, reacting to client input.

Key instructions:
- Ask relevant questions.
- Adapt tone to the client's knowledge.
- Avoid complex jargon.
- Conclude conversations naturally with an explicit `[END]` marker.

Excerpt:
> You are an expert financial advisor specializing in mortgages. Start by greeting the client and asking relevant, natural questions to understand their goal...

---

## Reproducibility

To generate your own dataset:
1. Install `wizardSdata`
2. Prepare your profiles and templates
3. Use the generation script:

```python
import wizardsdata as wsd

wsd.set_config(
  API_KEY="your_openai_key",
  template_client_prompt="templates/financial_client_01.j2",
  template_advisor_prompt="templates/financial_advisor_01.j2",
  file_profiles="profiles/retail_banking_160.json",
  file_output="outputs/retail_banking_dataset.json",
  model_client="gpt-4o",
  model_advisor="gpt-4o",
  temperature_client=0.6,
  temperature_advisor=0.2,
  max_recommended_questions=10
)

wsd.start_generation()
```

📚 For full documentation: [WizardSData Docs](https://peremartra.github.io/WizardSData)

---

## Dataset License

MIT License.

---

## Author

Created by **Pere Martra** as part of the FinLLMOpt initiative.

For questions, suggestions, or contributions: feel free to open an issue or contact me via [LinkedIn](https://www.linkedin.com/in/pere-martra/).

