import os
os.environ["TRANSFORMERS_CACHE"] = "cache/"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import transformers
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import torch
import time
import math

device = "cuda:0"

model_class = "LLAMA30_instruct"

print("model_class: ", model_class)

model2path = {"WizardVicunaLLAMA30":  "../Wizard-Vicuna-30B-Uncensored-fp16/",
              "MPT30": "../mpt-30b",
              "LLAMA2_70": "../Llama-2-70b-hf",
              'LLAMA2_13_chat': "../Llama-2-13b-chat-hf",
              'LLAMA30_instruct': "../llama-30b-instruct-2048",
              "Falcon40_instruct": "../falcon-40b-instruct",
              "WizardFalcon40": "../WizardLM-Uncensored-Falcon-40b"}

checkpoint = model2path[model_class]

def select_opt_id(idx):
    flag = False
    for id in idx:
        if flag:
            return id
        else:
            token = tokenizer.decode(id)
            if token == "(":
                flag = True
    return idx[0]

init_time = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir="cache/", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
final_time = time.perf_counter()
print("tokenizer loading time: ", final_time - init_time)

init_time = time.perf_counter()

Bidx = tokenizer.encode("(B)")
Aidx = tokenizer.encode("(A)")

Aid = select_opt_id(Aidx)
Bid = select_opt_id(Bidx)

model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                             cache_dir="cache/",
                                             device_map="auto",
                                             torch_dtype=torch.float16,
                                             revision=True,
                                             trust_remote_code=True)

model.half()
final_time = time.perf_counter()
print("model loading time: ", final_time - init_time)
print(model.hf_device_map)

model.eval()
model = torch.compile(model)

question_prompts = ["2+2=4.\nIs the last reasoning step correct?\n(A) Yes\n(B) No\n",
                    "2+2=5.\nIs the last reasoning step correct?\n(A) Yes\n(B) No\n",
                    "2+2=4.\nIs the last reasoning step incorrect?\n(A) Yes\n(B) No\n",
                    "2+2=5.\nIs the last reasoning step incorrect?\n(A) Yes\n(B) No\n",
                    "2+2=4.\nIs the last calculation step correct?\n(A) Yes\n(B) No\n",
                    "2+2=5.\nIs the last calculation step correct?\n(A) Yes\n(B) No\n",
                    "2+2=4.\nIs the last calculation step incorrect?\n(A) Yes\n(B) No\n",
                    "2+2=5.\nIs the last calculation step incorrect?\n(A) Yes\n(B) No\n",
                    "2+2=4.\nIs the last calculation/reasoning step correct?\n(A) Yes\n(B) No\n",
                    "2+2=5.\nIs the last calculation/reasoning step correct?\n(A) Yes\n(B) No\n",
                    "2+2=4.\nIs the last calculation/reasoning step incorrect?\n(A) Yes\n(B) No\n",
                    "2+2=5.\nIs the last calculation/reasoning step incorrect?\n(A) Yes\n(B) No\n",
                    "Given: P, P => Q. Therefore: Q.\nIs the last reasoning step correct?\n(A) Yes\n(B) No\n",
                    "Given: Q, P => Q. Therefore: P.\nIs the last reasoning step correct?\n(A) Yes\n(B) No\n",
                    "Given: P, P => Q. Therefore: Q.\nIs the last reasoning step incorrect?\n(A) Yes\n(B) No\n",
                    "Given: Q, P => Q. Therefore: P.\nIs the last reasoning step incorrect?\n(A) Yes\n(B) No\n",
                    "Given: P, P => Q. Therefore: Q.\nIs the last calculation step correct?\n(A) Yes\n(B) No\n",
                    "Given: Q, P => Q. Therefore: P.\nIs the last calculation step correct?\n(A) Yes\n(B) No\n",
                    "Given: P, P => Q. Therefore: Q.\nIs the last calculation step incorrect?\n(A) Yes\n(B) No\n",
                    "Given: Q, P => Q. Therefore: P.\nIs the last calculation step incorrect?\n(A) Yes\n(B) No\n",
                    "Given: P, P => Q. Therefore: Q.\nIs the last calculation/reasoning step correct?\n(A) Yes\n(B) No\n",
                    "Given: Q, P => Q. Therefore: P.\nIs the last calculation/reasoning step correct?\n(A) Yes\n(B) No\n",
                    "Given: P, P => Q. Therefore: Q.\nIs the last calculation/reasoning step incorrect?\n(A) Yes\n(B) No\n",
                    "Given: Q, P => Q. Therefore: P.\nIs the last calculation/reasoning step incorrect?\n(A) Yes\n(B) No\n",
                    ]

answer_prompt = "The answer is ("

prompt_templates = []
if model_class == "WizardVicunaLLAMA30":
    for question_prompt in question_prompts:
        prompt_template = f"USER: {question_prompt}\nASSISTANT: {answer_prompt}"
        prompt_templates.append(prompt_template)
elif model_class == "MPT30":
    for question_prompt in question_prompts:
        prompt_templates.append(question_prompt + answer_prompt)
elif model_class == "LLAMA2_70":
    for question_prompt in question_prompts:
        prompt_templates.append(question_prompt + answer_prompt)
elif model_class == "LLAMA2_13_chat":
    for question_prompt in question_prompts:
        system = "You are an AI assistant that helps people find information. " \
                 "User will you give you a question. Your task is to answer as faithfully as you can. "
        prompt_template = f"SYSTEM: {system}\nUSER: {question_prompt}\nASSISTANT: {answer_prompt}"
        prompt_templates.append(prompt_template)
elif model_class == "LLAMA30_instruct":
    for question_prompt in question_prompts:
        system = "You are an AI assistant that helps people find information. " \
                 "User will you give you a question. Your task is to answer as faithfully as you can. "
        prompt_template = f"{system}\n\n### Instruction: {question_prompt}\n\n### Response: {answer_prompt}"
        prompt_templates.append(prompt_template)
elif model_class == "Falcon40_instruct":
    for question_prompt in question_prompts:
        prompt_template = f"User: {question_prompt}\nAssistant: {answer_prompt}"
        prompt_templates.append(prompt_template)
elif model_class == "WizardFalcon40":
    for question_prompt in question_prompts:
        prompt_template = f"{question_prompt}\n### Response: {answer_prompt}"
        prompt_templates.append(prompt_template)


with torch.no_grad():
    for prompt_template in prompt_templates:
        #print("prompt_template: ", prompt_template)
        inputs = tokenizer(prompt_template, return_tensors="pt", max_length=1024, truncation=True, padding=True)
        input_ids = inputs["input_ids"].to(device)
        init_time = time.perf_counter()
        output = model.generate(input_ids=input_ids,
                                max_new_tokens=100,
                                bos_token_id=tokenizer.bos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.eos_token_id,
                                output_scores=True,
                                do_sample=False,
                                return_dict_in_generate=True,
                                use_cache=True)
        #print("output: ", output)
        scores = output.scores[0][0]
        A_score = math.exp(scores[Aid].item())
        B_score = math.exp(scores[Bid].item())
        Aprob = 0 if A_score == 0 else A_score / (A_score + B_score)
        Bprob = 0 if B_score == 0 else B_score / (A_score + B_score)
        print("sequence: ", tokenizer.decode(output.sequences[0]))
        print("Aprob: ", Aprob)
        print("Bprob: ", Bprob)
        final_time = time.perf_counter()
        print("Generation Time: ", final_time - init_time)
        print("\n\n")

