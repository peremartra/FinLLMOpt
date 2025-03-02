# FinLLMOpt
FinLLMOpt is a comprehensive collection of optimized Financial Large Language Models (FinLLMs) designed to address various tasks in the financial domain. This repository aims to provide efficient and effective NLP models tailored for financial applications, facilitating tasks such as sentiment analysis, risk assessment, and more.

## Overview
In the rapidly evolving financial sector, the need for specialized NLP models has become paramount. FinLLMOpt addresses this need by offering a suite of models fine-tuned and optimized for various financial tasks. These models serve as a laboratory to test optimizations and advanced pruning methods, pushing the boundaries of efficiency and performance. Each model is designed with efficiency in mind, ensuring deployment feasibility on both high-end servers and modern edge devices.

## Repository Structure. 
* models: Contains directories for each model, with individual README.md files and associated scripts.

## Available Models. 
### [Llama-Finsent-S](https://github.com/peremartra/FinLLMOpt/tree/main/Llama-FinSent-S). 
[![Download Model](https://img.shields.io/badge/Download%20Model-Hugging%20Face-blue?style=for-the-badge&logo=HuggingFace)](https://huggingface.co/oopere/Llama-FinSent-S)

Llama-FinSent-S is a fine-tuned and pruned version of LLaMA-3.2-1B, specifically designed for financial sentiment analysis. It is 26% smaller than the original model while achieving a 50% improvement in the IFEVAL benchmark and an impressive 400% improvement in the MUSR benchmark. These benchmarks measure the model's ability to follow instructions and reason, which are crucial for effective financial sentiment analysis.

### [FinChat-XS](https://github.com/peremartra/FinLLMOpt/tree/main/FinChat-XS)
[![Download Model](https://img.shields.io/badge/Download%20Model-Hugging%20Face-blue?style=for-the-badge&logo=HuggingFace)](https://huggingface.co/oopere/FinChat-XS)
FinChat-XS is a lightweight financial domain language model (360M) designed to answer questions about finance, markets, investments, and economics in a conversational style. This first version of the model has been trained using the dataset [Sujet-Finance-Instruct-177k](sujet-ai/Sujet-Finance-Instruct-177k). However, the need for high-quality conversational datasets focused on personal finance has emerged. This repository will work towards addressing that gap, aiming to enhance the quality of financial chat models.

## Follow Model-Specific Instructions. 
Each model has it's own README.md with detailed usage instructions. 

## Contributing. 
Contributions are welcome. Yo can contribute in any wany, if you want to fix an issue a typo or just add some inforation fork the repository and commit your changes with a pull request. 
You can also contribute to the repository getting responsability in the creation of a future model or a future work. 
1. See the list of future works.
2. Select one of them to contribute.
3. Open a Issue or a Discussion.
4. The task will be assigned to you.
5. We can assist you if you need help.
6. Fork the repository.
7. Create a new branch.
8. Commit your changes.
9. Push the Branch.
10. open a Pull Request.
11. Be promoted in my social networks as important contibutor to the repository :-)

## License.
This project is licensed under a permisive Apache-2.0 License. See the LICENSE file for details. However, some models derived from LLaMA models or other sources may have more restrictive licenses. Please check individual model directories for specific licensing terms.

## Citation
```bibtex
@misc{FinLLMOpt2025,
  title={FinLLMOpt: Optimized Financial Large Language Models},
  author={Pere Martra},
  year={2025},
  url={https://github.com/your_username/FinLLMOpt}
}
```
