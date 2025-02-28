import argparse
import random

from longproc.longproc_data import load_longproc_data
from openai import OpenAI
from tqdm import tqdm

try:
    import torch
    from vllm import LLM, SamplingParams, TokensPrompt
    from transformers import AutoTokenizer
except ImportError:
    pass

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="html_to_tsv_0.5k")
    parser.add_argument("--path", type=str, default="./data", help="Path to data")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18", help="Model")

    parser.add_argument("--test_loading", action="store_true", help="Test loading data")

    return parser.parse_args()

def test_loading_all():
    def test_loading(dataset):
        data, eval_func = load_longproc_data(dataset, "./data")
        print(f"Dataset: {dataset}")
        print(f"N samples: {len(data)}")
        print(f"Eval func: {eval_func}")
        print(f"Max input chars: {max([len(d['input_prompt']) for d in data])}")
        print(f"Max output chars: {max([len(d['reference_output']) for d in data])}")
    [test_loading(d) for d in ["path_traversal_0.5k", "path_traversal_2k", "path_traversal_8k"]]

    [test_loading(d) for d in ["html_to_tsv_0.5k", "html_to_tsv_2k", "html_to_tsv_8k"]]

    [test_loading(d) for d in ["pseudo_to_code_0.5k", "pseudo_to_code_2k",]]

    [test_loading(d) for d in ["travel_planning_2k", "travel_planning_8k"]]

    [test_loading(d) for d in ["tom_tracking_0.5k", "tom_tracking_2k", "tom_tracking_8k"]]

    [test_loading(d) for d in ["countdown_0.5k", "countdown_2k", "countdown_8k"]]


def query_openai(model: str, user_prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return completion.choices[0].message.content


class _VLLMBackend:
    llm = None
    tokenizer = None
    init_args = None # model_name, max_model_len

def query_hf(model: str, user_prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    if _VLLMBackend.llm is None:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True,)
        _VLLMBackend.llm = LLM(model=model, tensor_parallel_size=torch.cuda.device_count(), dtype="auto",)
        _VLLMBackend.tokenizer = tokenizer

    llm = _VLLMBackend.llm
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
    tokenizer = _VLLMBackend.tokenizer

    prompt = [{"role": "user", "content": user_prompt}]
    # NOTE: pass in token ids which we find returns better results closer to hf generate
    token_prompt = TokensPrompt(prompt_token_ids=tokenizer.apply_chat_template(conversation=prompt, add_generation_prompt=True, tokenize=True,))

    outputs = llm.generate(
        prompts=[token_prompt],
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    generated_text = outputs[0].outputs[0].text

    return generated_text


def main():
    args = _parse_args()

    if args.test_loading:
        test_loading_all()
        return

    random.seed(args.seed)

    # allows some buffer to accomdate variations in token usage for different tokenizers
    if args.max_tokens is None:
        if "0.5k" in args.dataset:
            args.max_tokens = 1024
        elif "2k" in args.dataset:
            args.max_tokens = 3072
        elif "8k" in args.dataset:
            args.max_tokens = 9216


    dataset, eval_func = load_longproc_data(args.dataset, args.path)
    random.shuffle(dataset)
    if args.n_samples is not None:
        dataset = dataset[:args.n_samples]

    eval_metrics = []
    num_inspect = 2
    for i, d in tqdm(list(enumerate(dataset[:args.n_samples]))):
        if i < num_inspect:
            print(f"Sample {i+1}/{args.n_samples}")
            print(f"Prompt: {d['input_prompt']}")
            print(f"Reference: {d['reference_output']}")

        if "gpt" in args.model:
            prediction = query_openai(args.model, d["input_prompt"], args.max_tokens, args.temperature, args.top_p)
        else:
            prediction = query_hf(args.model, d["input_prompt"], args.max_tokens, args.temperature, args.top_p)

        metrics, additional_info = eval_func(prediction, d)
        if i < num_inspect:
            print(f"Prediction: {prediction}")
            print(f"Metrics: {metrics}")
            print(f"Additional info: {additional_info}")
        eval_metrics.append(metrics)

    for k, v in metrics.items():
        print(f"{k}: {sum([m[k] for m in eval_metrics])/len(eval_metrics)}")

if __name__ == '__main__':
    main()
