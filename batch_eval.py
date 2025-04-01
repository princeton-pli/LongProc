import argparse

import os
import numpy as np
import random
import tiktoken
import json

from cached_query_tool import cached_batch_query
from longproc.longproc_data import load_longproc_data
from datasets import Dataset as HFDataset

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)

    # data quantity
    parser.add_argument("--dataset", type=str, default="tom_tracking_0.5k")
    parser.add_argument("--path", type=str, default="./data", help="Path to data")
    parser.add_argument("--n_samples", type=int, default=105, help="Number of samples")

    # query args
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_model_len", type=int, default=128000)

    # control args
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--disable_cache", dest="enable_cache", action="store_false", default=True)


    args = parser.parse_args()

    if args.output_dir is None:
        saving_name = args.model.replace("/", "-")
        args.output_dir = os.path.join("results", saving_name)
    return args


def output_filename_func(args):
    # save results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_filename = f"{args.dataset}_max{args.max_tokens}t{args.temperature}p{args.top_p}_{args.seed}.json"
    return os.path.join(args.output_dir, output_filename)

# use this way to sample subset to make it complt with the original sample in the paper evaluation
def subsample_dataset(dataset, n_samples, seed=42):
    if not isinstance(dataset, HFDataset) and isinstance(dataset, list):
        dataset = HFDataset.from_list(dataset)
    dataset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))
    return dataset

def load_longproc_local_hf_dataset(dataset,):
    print("using local hf dataset")
    import datasets
    from longproc.longproc_data import load_long_proc_eval_func
    # load dataset
    hf_dataset = datasets.load_dataset("json", data_files=f"hf_datasets/{dataset}.jsonl",)["train"]
    # load eval function
    eval_func = load_long_proc_eval_func(dataset)
    return hf_dataset, eval_func

def main():
    args = _parse_args()
    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset, eval_func = load_longproc_data(args.dataset, args.path)
    # dataset, eval_func = load_longproc_local_hf_dataset(args.dataset)

    if args.n_samples is not None:
        # use this way to sample subset to make it complt with the original sample in dataset
        dataset = subsample_dataset(dataset, args.n_samples, args.seed)

    prompts = []
    for ex in dataset:
        prompts.append([{"role": "user", "content": ex["input_prompt"]}])

    cache_file = "caches/" + args.model.replace("/", "-") + ".sqlite"
    # batch querying
    outputs = cached_batch_query(cache_file, prompts, args.model, args.max_tokens, args.temperature, args.top_p, args.max_model_len, aux_kwargs={"enable_cache": args.enable_cache})


    all_metrics = []
    saving_info = []
    for ex, output in zip(dataset, outputs):
        assert len(output["output"]) == 1
        mets, _ = eval_func(output["output"][0], ex)
        all_metrics.append(mets)
        saving_info.append({
            "data": ex,
            "output": {"prompt": output["prompt"], "output": output["output"], "success": output["success"]},
            "metric": mets,
        })

    avg_metrics = {k: np.mean([x[k] for x in all_metrics]) * 100 for k in all_metrics[0].keys()}
    print([f"{k}: {v:.1f}" for k, v in avg_metrics.items()])
    output_filename = output_filename_func(args)

    # with open(output_filename, "w") as f:
        # for ex, prompt, output in zip(dataset, prompts, outputs):
    avg_metrics = {k: np.mean([x[k] for x in all_metrics]) for k in all_metrics[0].keys()}
    output_content = {
        "args": args.__dict__,
        "saving_info": saving_info,
        "avg_metrics": avg_metrics
    }
    with open(output_filename, "w") as f:
        json.dump(output_content, f, indent=2)

if __name__=="__main__":
    main()
