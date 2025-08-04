# JT-Math: A Family of Open-Source Models for Advanced Mathematical Reasoning

<!-- <p align="center">
    <a href="https://www.arxiv.org/abs/2507.19748" target="_blank">
        <img src="https://img.shields.io/badge/Paper-ArXiv-red">
    </a>
    <a href="https://huggingface.co/JT-LM" target="_blank">
        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue">
    </a>
    🤖 <a href="https://modelscope.cn/">ModelScope
</p> -->

<p align="center">
    <a href="<PAPER_LINK_PLACEHOLDER>" target="_blank">
        <img src="https://img.shields.io/badge/Paper-ArXiv-red">
    </a>
    <a href="https://huggingface.co/JT-LM" target="_blank">
        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue">
    </a>
    <a href="https://www.modelscope.cn/organization/JiuTian-AI" target="_blank">
        <img src="https://img.shields.io/badge/%F0%9F%A4%96%20ModelScope-Models-blue">
    </a>
</p>


## Introduction



We are excited to  unveil **JT-Math**, a powerful, open-source family of 8-billion parameter large language models specifically engineered to advance the state-of-the-art in mathematical reasoning. The JT-Math series is designed to provide both strong foundational models and highly capable instruction-tuned models for a wide range of mathematical tasks, from basic problem-solving to complex, multi-step reasoning.

This repository serves as the official hub for the JT-Math series, providing all the necessary code, model links, and comprehensive documentation to get you started.

------



## Highlights



The JT-Math family boasts three distinct models, each meticulously optimized for a specific purpose:

- 🧮 **JT-Math-8B-Base**: The foundational powerhouse of the series. This model was meticulously developed through a comprehensive three-stage pre-training process on a high-quality, 210 billion token corpus. It supports an impressive native context window of **32,768 tokens**, making it ideal for deep dives into mathematical concepts.
- 🗣️ **JT-Math-8B-Instruct**: Your go-to model for versatile instruction-following. Fine-tuned from the base model using a synergistic combination of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), it truly excels at solving mathematical problems presented in conversational or instructional formats within an **8K token window**.
- 🧠 **JT-Math-8B-Thinking**: Our premier model for tackling the most challenging mathematical problems. Featuring an extended **32,768-token context window** and optimized with an advanced multi-stage curriculum learning RL pipeline, JT-Math-8B-Thinking is engineered to enable deep, multi-step reasoning.

------




## Performance
The JT-Math models, particularly **JT-Math-8B-Thinking**, consistently achieve **state-of-the-art performance** across a range of key mathematical reasoning benchmarks. They proudly outperform other leading open-source models in the ~8B parameter class, demonstrating their superior capabilities.
Below is a summary of our evaluation results. For a more in-depth analysis and detailed performance metrics, we encourage you to refer to our technical report.

![alt text](<Evaluation Results.png>)





## Model Zoo



We are committed to fostering open science and are delighted to release all three JT-Math models to the community under an open-source license.

| Model Name          | Context Length | Hugging Face Link                                          | ModelScope Link                                            | Notes                                                      |
| ------------------- | -------------- | ---------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------- |
| JT-Math-8B-Base     | 32K            |  [Link](https://huggingface.co/JT-LM/JT-Math-8B-Base)     |  [Link](https://www.modelscope.cn/models/JiuTian-AI/JT-Math-8B-Base) | The foundational base model. Ideal for custom fine-tuning. |
| JT-Math-8B-Instruct | 32K            |  [Link](https://huggingface.co/JT-LM/JT-Math-8B-Instruct) |  [Link](https://www.modelscope.cn/models/JiuTian-AI/JT-Math-8B-Instruct) | Instruction-tuned for general math problem-solving.        |
| JT-Math-8B-Thinking | 32K            |  [Link](https://huggingface.co/JT-LM/JT-Math-8B-Thinking) |  [Link](https://www.modelscope.cn/models/JiuTian-AI/JT-Math-8B-Thinking) | The premier model for complex, long-context reasoning.     |
------



## How to Get Started

### 1. JT-Math-8B-Base


```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "JT-LM/JT-Math-8B-Base"

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


```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "JT-LM/JT-Math-8B-Instruct"

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


```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "JT-LM/JT-Math-8B-Thinking"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

prompt = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
messages = [
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

gen_kwargs = {
    "do_sample": True,
    "temperature": 0.65,
    "max_new_tokens": 32768,
}
generated_ids = model.generate(
    **model_inputs,
    **gen_kwargs
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

raw_content = tokenizer.decode(output_ids, skip_special_tokens=True)
if "</think>" in raw_content:
    thinking_content = raw_content.rsplit("</think>", 1)[0].strip("\n")
    content = raw_content.rsplit("</think>", 1)[1].strip("\n")
else:
    thinking_content = raw_content
    content = ""

print("raw content:", raw_content)
print("thinking content:", thinking_content)
print("content:", content)
```

------



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

