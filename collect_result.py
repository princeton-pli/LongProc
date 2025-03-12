import os
import sys
import json
import pathlib
import yaml

import re
import argparse


def collect_results(args, model, benchmark, max_tokens=".*"):

    wildcard = f"{benchmark}_max{max_tokens}.*.json"
    filenames = os.listdir(f"{args.output_dir}/{model}")
    filenames = [f for f in filenames if re.match(wildcard, f)]

    if len(filenames) == 0:
        print(filenames)
        print(wildcard)
        data = {}
    elif len(filenames) > 1:
        print(model)
        print(f"Found {len(filenames)} files matching {wildcard}")
        print([os.path.join(args.output_dir, model, f) for f in filenames])
        raise RuntimeError(f"Found {len(filenames)} files matching {wildcard}")
    # assert len(filenames) == 1, f"Found {len(filenames)} files matching {wildcard}"
    else:
        filename = filenames[0]
        with open(f"{args.output_dir}/{model}/{filename}", "r") as f:
            data = json.load(f)["avg_metrics"]

    return data


_MAIN_METRICS = ["substring_exact_match", "accuracy", "f1", "NDCG@10"]
def print_table(results: dict, met: str):
    """print a csv table of results"""
    # header
    rows = []
    rows.append([met] + list(list(results.values())[0].keys()) + ["avg"])
    for model in results:
        row = [model]
        all_raw_scores = []
        for benchmark in results[model]:
            if met == "main":
                dp = None
                for m in _MAIN_METRICS:
                    if m in results[model][benchmark]:
                        dp = results[model][benchmark][m]
                        break
            else:
                dp = results[model][benchmark].get(met, None)
            all_raw_scores.append(dp)

        valid_scores = [dp for dp in all_raw_scores if dp is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
        all_raw_scores.append(avg_score)

        for dp in all_raw_scores:
            if met in ["main",]:
                dp = dp * 100 if dp is not None else dp
                dp = f"{dp:.1f}" if dp is not None else dp
            row.append(dp)
        rows.append(row)
    # print
    for row in rows:
        print(",".join([str(x) for x in row]))



_MODELS = [
    "vllm-models-Llama-3.1-8B-Instruct-",
    "vllm-models-DeepSeek-R1-Distill-Llama-8B-",
    "vllm-models-Qwen2.5-32B-Instruct-",
    "vllm-models-s1-32B-",
    "vllm-models-s1.1-32B-",
    "vllm-models-DeepSeek-R1-Distill-Qwen-32B-",
    "vllm-models-Llama-3.1-70B-Instruct-",
    "vllm-models-DeepSeek-R1-Distill-Llama-70B-"
]


_BENCHMARKS = [
    "html_to_tsv_0.5k",
    "html_to_tsv_2k",
    "html_to_tsv_8k",
    "pseudo_to_code_0.5k",
    "pseudo_to_code_2k",
    "path_traversal_0.5k",
    "path_traversal_2k",
    "path_traversal_8k",
    "tom_tracking_0.5k",
    "tom_tracking_2k",
    "tom_tracking_8k",
    "countdown_0.5k",
    "countdown_2k",
    "countdown_8k",
    "travel_planning_2k",
    "travel_planning_8k",
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--filter", type=str, default=None)

    args = parser.parse_args()

    # models = []
    # if args.model is None:
    #     models = os.listdir(args.output_dir)
    #     if args.filter is not None:
    #         models = [m for m in models if args.filter in m]
    #     models.sort()
    # else:
    models = _MODELS


    benchmarks = _BENCHMARKS

    # get a giant table of results; results dict results[model][benchmark][metrics]
    results = {}
    for model in models:
        results[model] = {}
        for bench in benchmarks:
            results[model][bench] = collect_results(args, model, bench,)

    # accuracy table
    print_table(results, "main")
