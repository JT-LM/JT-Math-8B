# JT-Math: A Family of Open-Source Models for Advanced Mathematical Reasoning 


<p align="center">
    <a href="<PAPER_LINK_PLACEHOLDER>" target="blank">
        <img src="https://img.shields.io/badge/Paper-ArXiv-red">
    <a href="[https://huggingface.co/JT-LM/JT-Math-8B-Thinking](https://huggingface.co/JT-LM)" target="blank">
        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue">
    <a href="./LICENSE" target="blank">
        <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg">
</p>







## Introduction

We are excited to introduce JT-Math, a powerful, open-source family of 8-billion parameter large language models specifically engineered to advance the state-of-the-art in mathematical reasoning. The JT-Math series is designed to provide strong foundational models and highly capable instruction-tuned models for a wide range of mathematical tasks, from basic problem-solving to complex, multi-step reasoning.

This repository contains the official code, model links, and documentation for the JT-Math series.



## Highlights

The JT-Math family includes three distinct models, each optimized for a specific purpose:

- JT-Math-8B-Base: The foundational model of the series. It was developed through a comprehensive three-stage pre-training process on a high-quality, 210 billion token corpus and supports a native context window of 32,768 tokens.
- JT-Math-8B-Instruct: A versatile instruction-following model. Fine-tuned from the base model with a combination of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), it excels at solving mathematical problems presented in a conversational or instructional format within an 8K token window.
- JT-Math-8B-Thinking:  It features an extended 32,768-token context window and is optimized with an advanced multi-stage curriculum learning RL pipeline, enabling it to tackle difficult mathematical challenges that require deep, multi-step reasoning.



## Performance

The JT-Math models, particularly `JT-Math-8B-Thinking`, achieve state-of-the-art performance across a range of key mathematical reasoning benchmarks, outperforming other open-source models in the ~8B parameter class.

Below is a summary of our evaluation results. For more details, please refer to our technical report.





## Model Zoo

We release all three models to the community under an open-source license.

| Model Name            | Context Length | Hugging Face Link                                          | Notes                                                      |
| --------------------- | -------------- | ---------------------------------------------------------- | ---------------------------------------------------------- |
| `JT-Math-8B-Base`     | 32K            | [🤗 Link](https://huggingface.co/JT-LM/JT-Math-8B-Base)     | The foundational base model. Ideal for custom fine-tuning. |
| `JT-Math-8B-Instruct` | 32K            | [🤗 Link](https://huggingface.co/JT-LM/JT-Math-8B-Instruct) | Instruction-tuned for general math problem-solving.        |
| `JT-Math-8B-Thinking` | 32K            | [🤗 Link](https://huggingface.co/JT-LM/JT-Math-8B-Thinking) | The premier model for complex, long-context reasoning.     |



## How to Get Started

### 1. JT-Math-8B-Base

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Jiutian/JT-Math-8B-Base"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

prompt = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
text = f"Question:\n{prompt}\nAnswer:\n"
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

gen_kwargs = {
    "do_sample": False,
    "max_new_tokens": 8192,
}
generated_ids = model.generate(
    **model_inputs,
    **gen_kwargs
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

response = tokenizer.decode(output_ids, skip_special_tokens=True)
print("response:", response)
```



### 2. JT-Math-8B-Instruct

This model is fine-tuned to follow instructions and solve problems in a conversational format.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Jiutian/JT-Math-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

prompt = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

gen_kwargs = {
    "do_sample": False,
    "max_new_tokens": 8192,
}
generated_ids = model.generate(
    **model_inputs,
    **gen_kwargs
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

response = tokenizer.decode(output_ids, skip_special_tokens=True)
print("response:", response)
```



### 3. JT-Math-8B-Thinking

This is our most capable model, designed for complex, multi-step problems.

Python

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "your-hf-repo/JT-math-8B-Thinking"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Example of a more complex word problem
problem = (
    "A farm has chickens and rabbits. When the farmer counts the heads, he gets a total of 50. ""When he counts the legs, he gets a total of 140. How many chickens and how many rabbits are on the farm? ""Show your work step-by-step."
)
messages = [{"role": "user", "content": problem}]

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

outputs = model.generate(inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```



## Citation



If you find our work useful for your research, please consider citing our paper:

```latex
@article{JT-math2025,
  title={JT MATH: A MULTI-STAGE FRAMEWORK FOR ADVANCED MATHEMATICAL REASONING IN LARGE LANGUAGE MODELS},
  author={Your Authors},
  journal={Your Journal/Conference (e.g., arXiv preprint arXiv:xxxx.xxxxx)},
  year={2025}
}
```



## License

This project is licensed under the [Your License Name, e.g., Apache 2.0 License]. Please see the `LICENSE` file for details.
