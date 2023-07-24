from vllm import SamplingParams, LLM
from transformers import AutoTokenizer, LlamaTokenizer
from typing import List, Optional, Union


class generator:
    def __init__(self, model_name: str,
                 tensor_parallel_size: int):

        self.AutoTokenizer = AutoTokenizer
        use_fast = True
        if model_name == "LLAMA30_instruct":
            checkpoint = "../llama-30b-instruct-2048"
        elif model_name == "LLAMA60_instruct":
            checkpoint = "../llama-65b-instruct"
        elif model_name == "Redmond":
            use_fast = False
            checkpoint = "../Redmond-Puffin-13B"
        else:
            raise ValueError("Model named '{}' not supported.".format(model_name))

        self.model = LLM(checkpoint, tensor_parallel_size=tensor_parallel_size, swap_space=16)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir="cache/", use_fast=use_fast)
        self.model.llm_engine.tokenizer = self.tokenizer


    def generate(self, prompt: List[str],
                 temperature: float = 0,
                 max_length: int = 256,
                 num_samples: int = 1,
                 top_k: int = -1,
                 stop: Union[str, List[str]] = [],
                 logprobs: Optional[int] = None):

        stop = stop + [self.tokenizer.eos_token]
        sampling_params = SamplingParams(temperature=temperature,
                                         max_tokens=max_length,
                                         n=num_samples,
                                         top_k=top_k,
                                         stop=stop,
                                         logprobs=logprobs)
        return self.model.generate(prompt, sampling_params)
