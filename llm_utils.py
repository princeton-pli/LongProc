from typing import (
    List,
    Any
)
import os
import pickle

import logging
import time

from functools import partial
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _call_api(func, limit=5, pause=10):
    count = 0
    while True:
        try:
            output = func()
            break
        except Exception as e:
            logger.info(f"Exception while using api: {e}")
            if "rate limit" in str(e).lower() or "rate_limit" in str(e).lower() or "quota" in str(e).lower() or "429" in str(e):
                logger.info(f"Rate limit exceeded, waiting {pause} secs and retrying...")
                time.sleep(pause)
            elif count < limit:
                logger.info(f"Encountered error {e}, retrying...")
                count += 1
            else:
                logger.info("Skipping generation due to unknown error")
                raise e
    return output

def key_of_llm_query(*args, **kwargs) -> str:
    sorted_kwargs = sorted(kwargs.items())
    key = pickle.dumps((args, tuple(sorted_kwargs)))
    return key


# NOTE: ONLY return 1 output per prompt
def _batch_openai_query(
        prompts: List[Any],
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        max_model_len: int = None, # not needed for openai
        query_kwargs: dict = {},
        aux_kwargs: dict = {},
    ):

    # TODO: not actually batching yet
    import openai
    client = openai.OpenAI()

    canonical_outputs = []
    for prompt in tqdm(prompts):
        func = partial(
            client.chat.completions.create, 
            model=model_name, 
            messages=prompt, 
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **query_kwargs
        )
        completion = _call_api(func)
        content = completion.choices[0].message.content
        canonical_outputs.append({"model": model_name, "prompt": prompt, "output": content, "success": True,})
    return canonical_outputs

class _VLLMBackend:
    llm = None
    tokenizer = None
    init_args = None # model_name, max_model_len

# NOTE: ONLY return 1 output per prompt
def _batch_vllm_query(
        prompts: List[Any],
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        max_model_len: int = None,
        query_kwargs: dict = {}, # TODO: not using query_kwargs yet
        aux_kwargs: dict = {},
    ):
    assert model_name.startswith("vllm/")
    assert max_model_len is not None
    model_name = model_name[len("vllm/"):]

    import torch
    from vllm import LLM, SamplingParams, TokensPrompt
    if _VLLMBackend.llm is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)
        _VLLMBackend.llm = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count(),
            dtype="auto",  max_model_len=max_model_len, gpu_memory_utilization=aux_kwargs.get("vllm_gpu_memory_utilization", 0.95))
        _VLLMBackend.init_args = (model_name, max_model_len)
        _VLLMBackend.tokenizer = tokenizer

    assert _VLLMBackend.init_args == (model_name, max_model_len)
    llm = _VLLMBackend.llm
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
    tokenizer = _VLLMBackend.tokenizer

    token_prompts = []
    # NOTE: pass in token ids which we find returns better results closer to hf generate
    for prompt in prompts:
        token_prompts.append(TokensPrompt(prompt_token_ids=tokenizer.apply_chat_template(conversation=prompt, add_generation_prompt=True, tokenize=True,)))

    outputs = llm.generate(
        prompts=token_prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    canonical_outputs = []
    for prompt, output in zip(prompts, outputs):
        generated_text = output.outputs[0].text
        canonical_outputs.append({"model": model_name, "prompt": prompt, "output": generated_text, "success": True,})
    return canonical_outputs


# NOTE: ONLY return 1 output per prompt
def _batch_hf_query(
        prompts: List[Any],
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        max_model_len: int = None,
        query_kwargs: dict = {}, # TODO: not using query_kwargs yet
        aux_kwargs: dict = {},
    ):
    assert model_name.startswith("hf/")
    model_name = model_name[len("hf/"):]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto", trust_remote_code=True)

    all_outputs = []
    with torch.inference_mode():
        for prompt in tqdm(prompts):
            input_ids = tokenizer.apply_chat_template([prompt], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
            input_length = input_ids.shape[1]
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False if temperature == 0.0 else True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_answer = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            all_outputs.append({"model": model_name, "prompt": prompt, "output": gen_answer, "success": True,})

    return all_outputs


## OBSOLETE: using token ids gives better results than using chat prompt
def _batch_vllm_query_chat_prompt(
        prompts: List[Any],
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        max_model_len: int = None,
        query_kwargs: dict = {}, # TODO: not using query_kwargs yet
        aux_kwargs: dict = {},
    ):
    assert model_name.startswith("vllm/")
    assert max_model_len is not None
    model_name = model_name[len("vllm/"):]

    import torch
    from vllm import LLM, SamplingParams

    if _VLLMBackend.llm is None:
        _VLLMBackend.llm = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count(),
            dtype="auto",  max_model_len=max_model_len, gpu_memory_utilization=aux_kwargs.get("vllm_gpu_memory_utilization", 0.95))
        _VLLMBackend.init_args = (model_name, max_model_len)

    assert _VLLMBackend.init_args == (model_name, max_model_len)
    llm = _VLLMBackend.llm
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p)

    outputs = llm.chat(
        messages=prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    

    canonical_outputs = []
    for prompt, output in zip(prompts, outputs):
        generated_text = output.outputs[0].text
        canonical_outputs.append({"model": model_name, "prompt": prompt, "output": generated_text, "success": True,})
    return canonical_outputs


def batch_query(
        prompts: List[Any],
        model_name: str,
        gen_max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        max_model_len = None,
        query_kwargs: dict = {},
        aux_kwargs: dict = {},
    ):
    if model_name.startswith("gpt"):
        outputs = _batch_openai_query(prompts, model_name, gen_max_tokens, temperature, top_p, max_model_len, query_kwargs=query_kwargs, aux_kwargs=aux_kwargs)
    elif model_name.startswith("hf/"):
        outputs = _batch_hf_query(prompts, model_name, gen_max_tokens, temperature, top_p, max_model_len, query_kwargs=query_kwargs, aux_kwargs=aux_kwargs)
    elif model_name.startswith("vllm/"):
        outputs = _batch_vllm_query(prompts, model_name, gen_max_tokens, temperature, top_p, max_model_len, query_kwargs=query_kwargs, aux_kwargs=aux_kwargs)
    else:
        raise NotImplementedError

    assert len(outputs) == len(prompts)
    assert "model" in outputs[0] and "prompt" in outputs[0] and "output" in outputs[0] and "success" in outputs[0]

    return outputs
