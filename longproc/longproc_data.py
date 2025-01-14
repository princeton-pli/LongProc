from typing import Tuple, List, Dict, Callable

import json
import os
import yaml
import re
from .html_to_tsv_evaluator import evaluate_html_to_csv_compute_metrics
from .spoc_evaluator import evaluate_spoc_code
from .tom_tracking_evaluator import evaluate_tom_trace
from .countdown_evaluator import (
    build_countdown_demonstration,
    evaluate_countdown_final_solution,
    evaluate_countdown_search_procedure
)
from .travel_planning_evaluator import (
    build_travel_plan_demonstration,
    evaluate_travel_plan_solution,
    evaluate_travel_plan_search_procedure
)


def _extract_with_tag(response: str, tag: str):
    start = response.find(f"<{tag}>")
    end = response.find(f"</{tag}>")
    if start == -1 or end == -1:
        return None
    return response[start+len(tag)+2:end].strip()


def _load_html_to_tsv_data(dataset_name: str, path: str=None) -> Tuple[Dict, Callable]:
    assert dataset_name in ["html_to_tsv_0.5k","html_to_tsv_2k","html_to_tsv_8k"]
    if path is None: path = "longpro_data"

    path = os.path.join(path, "html_to_tsv")
    data_file = os.path.join(path, dataset_name + ".json")
    with open(data_file, "r") as f:
        data = json.load(f)

    with open(os.path.join(path, "prompts.yaml"), "r") as f:
        prompt = yaml.safe_load(f)
        user_prompt = prompt['USER_PROMPT']

    data_purged = []
    for d in data:
        task_id = d["task_id"]
        website_id = d["website_id"]
        task_topic = d["task_topic"]
        task_description = d["task_description"]
        ground_truth = d["gt"]
        tsv_header = d["tsv_header"]
        filtering_instruction = d["filtering_instruction"]

        html_str = open(os.path.join(path, d['html_path']), 'r').read()
        
        data_purged.append({
            "task_id": task_id,
            "website_id": website_id,
            "task_topic": task_topic,
            "task_description": task_description,
            "html_str": html_str,
            "reference_output": ground_truth,
            "tsv_header": tsv_header,
            "filtering_instruction": filtering_instruction
        })

    return {
        "data": data_purged,
        "prompt_template": user_prompt,
    }, eval_html_to_tsv

def eval_html_to_tsv(prediction: str, example: dict):
    """
    Returns: metrics (dict) and additional info to update the original sample with (dict)
    """
    # metric: f1, precision, recall, extraction_rate
    try:
        # prediction should be wrapped with ```tsv and ``` to be parsed
        search = re.search(r'```tsv([\s\S]*)```', prediction)
        # exactly one group should be matched
        if search is not None:
            prediction = search.group(1).strip()
        else:
            if "```tsv" in prediction:
                prediction = prediction.split("```tsv")[1]
            prediction = prediction.strip() # use all
    except Exception as e:
        ## if "TSV:" is not in the output, then return 0.0 for all metrics because the model didn't follow format
        return {"f1": .0, "precision": .0, "recall": .0,"extraction_rate": .0}, {"parsed_output": None,"error_msg": str(e)}
    eval_results = evaluate_html_to_csv_compute_metrics(prediction, example["reference_output"])
    return {"f1": eval_results["f1"], "precision": eval_results["precision"], "recall": eval_results["recall"],"extraction_rate": 1.0 if eval_results["error"] is None else 0.0}, {"parsed_output": prediction,"error_msg": eval_results["error"]}

def eval_pseudo_to_code(prediction: str, example: dict):
    """
    Returns: metrics (dict) and additional info to update the original sample with (dict)
    """

    ex_item = example["item"]
    # metric: accuracy, partial_accuracy (should be the same as accuracy), extraction_rate
    try:
        parsed_pred = re.sub(r'```c\+\+', '```cpp', prediction)
        parsed_pred = re.search(r'```cpp([\s\S]*)```', parsed_pred).group(1)
    except:
        parsed_pred = None
        return {"accuracy": .0, "partial_accuracy": .0, "extraction_rate": .0}, {"parsed_output": None,}

    result, error_msg = evaluate_spoc_code(parsed_pred, ex_item["testcases"])
    if result:
        return {"accuracy": 1.0, "partial_accuracy": 1.0, "extraction_rate": 1.0}, {"parsed_output": parsed_pred}
    else:
        return {"accuracy": 0.0, "partial_accuracy": 0.0, "extraction_rate": 1.0}, {"parsed_output": parsed_pred, "error_report": error_msg}


def _load_pseudo_to_code_data(dataset_name: str, path: str=None) -> Tuple[Dict, Callable]:
    assert dataset_name in ["pseudo_to_code_0.5k", "pseudo_to_code_2k"]
    if path is None: path = "longproc_data"

    path = os.path.join(path, "pseudo_to_code")

    data_file = os.path.join(path, dataset_name + ".json")
    with open(data_file, "r") as f:
        data = json.load(f)

    with open(os.path.join(path, "prompts.yaml"), "r") as f:
        prompt = yaml.safe_load(f)
        user_prompt = prompt['USER_PROMPT']

    data_purged = []
    for d in data:
        pseudocode = "\n".join(d["pseudocode_lines"])
        groud_truth = "\n".join(d["code_lines"])
        testcases = d["testcases"]
        problem_id = d["problem_id"]
        data_purged.append({
            "pseudocode": pseudocode,
            "reference_output": groud_truth,
            "testcases": testcases,
            "problem_id": problem_id,
        })

    return {
        "data": data_purged,
        "prompt_template": user_prompt,
    }, eval_pseudo_to_code


def eval_path_traversal(prediction: str, example: dict):
    """
    Returns: metrics (dict) and additional info
    """
    # metric: accuracy, partial_accuracy
    gt = example["reference_output"]
    parsed_pred = _extract_with_tag(prediction, "Route")
    if parsed_pred is None:
        return {"accuracy": .0, "partial_accuracy": .0, "extraction_rate": .0}, {"parsed_output": None, "error_report": "Parsing error"}

    gt = gt.strip()
    parsed_pred = parsed_pred.strip()
    if gt == parsed_pred:
        return {"accuracy": 1.0, "partial_accuracy": 1.0, "extraction_rate": 1.0}, {"parsed_output": parsed_pred, "error_report": None}

    gt_lines = gt.split("\n")
    pred_lines = parsed_pred.split("\n")

    error_report = None
    for i, (gl, pl) in enumerate(zip(gt_lines, pred_lines)):
        if gl != pl:
            error_report = {"line": i, "gt": gl, "pr": pl}
            break
    i += 1
    return {"accuracy": 0.0, "partial_accuracy": i/len(gt_lines), "extraction_rate": 1.0}, {"parsed_output": parsed_pred, "error_report": error_report}


def _load_path_traversal_data(dataset_name: str, path: str=None) -> Tuple[Dict, Callable]:
    # path is a dataset folder, containing different levels of path traversal data
    assert dataset_name in ["path_traversal_0.5k", "path_traversal_2k", "path_traversal_8k"]

    if path is None: path = "longproc_data"

    path = os.path.join(path, "path_traversal")

    data_file = os.path.join(path, dataset_name + ".json")
    with open(data_file, "r") as f:
        data = json.load(f)

    with open(os.path.join(path, "prompts.yaml"), "r") as f:
        prompt = yaml.safe_load(f)
        user_prompt = prompt['USER_PROMPT']

    data_purged = []
    for d in data:
        city_context = d["context_nl"]
        src_city = d["question_repr"][0]
        dst_city = d["question_repr"][1]
        data_purged.append({
            "city_context": city_context,
            "src_city": src_city,
            "dst_city": dst_city,
            "reference_output": d["answer_nl"],
        })

    return {
        "data": data_purged,
        "prompt_template": user_prompt,
    }, eval_path_traversal


def eval_tom_tracking(prediction: str, example: dict):
    gt_solution = example["reference_output"]

    parsed_pred = "\n".join([line for line in prediction.splitlines() if line.strip().startswith('-')])
    parsed_gt = "\n".join([line for line in gt_solution.splitlines() if line.strip().startswith('-')])

    strict_acc, partial_acc, error_report = evaluate_tom_trace(parsed_pred, parsed_gt)

    return {"accuracy": strict_acc, "partial_accuracy":  partial_acc, "extraction_rate": 1.0}, {"parsed_output": parsed_pred, "error_report": error_report}


def _load_tom_tracking_data(dataset_name: str, path: str=None) -> Tuple[Dict, Callable]:
    assert dataset_name in ["tom_tracking_0.5k", "tom_tracking_2k", "tom_tracking_8k"]

    if path is None: path = "longproc_data"

    path = os.path.join(path, "tom_tracking")

    data_file = os.path.join(path, dataset_name + ".json")
    with open(data_file, "r") as f:
        data = json.load(f)

    with open(os.path.join(path, "prompts.yaml"), "r") as f:
        prompt = yaml.safe_load(f)
        user_prompt = prompt['USER_PROMPT']

    data_purged = []
    for d in data:
        story_components = d["story_components"]
        story = d["story"]
        question = d["question"]
        reference_output = d["solution"]
        data_purged.append({
            "story_components": story_components,
            "story": story,
            "question": question,
            "reference_output": reference_output,
        })

    return {
        "data": data_purged,
        "prompt_template": user_prompt,
    }, eval_tom_tracking


def eval_countdown(prediction: str, example: dict):
    """
    Returns: metrics (dict) and additional info to update the original sample with (dict)
    """
    data_item = example["item"]
    pred_solution = _extract_with_tag(prediction, "Solution")
    nums = data_item["nums"]
    target = data_item["target"]
    if pred_solution is not None and evaluate_countdown_final_solution(nums, target, pred_solution):
            return {"accuracy": 1.0, "partial_accuracy": 1.0, "extraction_rate": 1.0}, {"parsed_output": pred_solution,}

    extraction_rate = 1.0 if pred_solution is not None else 0.0
    # handle probably unclosed search procedure
    if "# Search Procedure" not in prediction:
        return {"accuracy": 0.0, "partial_accuracy": 0.0, "extraction_rate": extraction_rate}, {"parsed_output": None,}

    pred_procedure = prediction.split("# Search Procedure")[-1].strip()

    ground_truth_procedure = data_item["reference_output"]
    # evaluate the procedure
    gt_procedure = ground_truth_procedure.split("# Search Procedure")[-1].split("Now we have found the target")[0].strip()

    partial_accuracy, error_report = evaluate_countdown_search_procedure(nums, target, pred_procedure, gt_procedure)
    return {"accuracy": 0.0, "partial_accuracy": partial_accuracy, "extraction_rate": extraction_rate}, {"parsed_output": pred_solution, "error_report": error_report}


def _load_countdown_data(dataset_name: str, path: str=None) -> Tuple[Dict, Callable]:
    assert dataset_name in ["countdown_0.5k", "countdown_2k", "countdown_8k"]

    if path is None: path = "longproc_data"

    path = os.path.join(path, "countdown")

    data_file = os.path.join(path, dataset_name + ".json")
    with open(data_file, "r") as f:
        data = json.load(f)

    with open(os.path.join(path, "prompts.yaml"), "r") as f:
        prompt = yaml.safe_load(f)
        user_prompt = prompt['USER_PROMPT']

    def build_icl_demonstration():
        _DEMO_SET = [
            {"nums": [40, 19, 23, 7], "target": 29,},
            {"nums": [9, 16, 6, 18], "target": 12,},
        ]
        examples = []
        for demo in _DEMO_SET:
            _, demonstration = build_countdown_demonstration(demo["nums"], demo["target"])
            examples.append(f"# Example\nNumbers: {demo['nums']}\nTarget: {demo['target']}\n\n{demonstration}")
        examples = "\n\n".join(examples)
        return examples

    # partially fill the user prompt
    user_prompt = user_prompt.format(demonstration=build_icl_demonstration(), nums="{nums}", target="{target}")

    data_purged = []
    for d in data:
        nums = d["nums"]
        target = d["target"]
        solution, demonstration = build_countdown_demonstration(nums[:], target)
        solution_str = "\n".join(solution)
        assert evaluate_countdown_final_solution(nums, target, solution_str), f"Failed to evaluate solution {solution_str}"
        data_purged.append({
            "nums": nums,
            "target": target,
            "solution": solution,
            "reference_output": demonstration,
        })


    return {
        "data": data_purged,
        "prompt_template": user_prompt,
    }, eval_countdown


def eval_travel_planning(prediction: str, example: dict):
    plan_text = _extract_with_tag(prediction, "Plan")
    data_item = example["item"]
    if plan_text is None:
        extraction_rate = 0.0
    else:
        extraction_rate = 1.0
        plan_text = plan_text.strip()

    if plan_text is not None:
        accuracy = evaluate_travel_plan_solution(data_item["ground_truth_cities"], data_item["ground_truth_durations"], plan_text)
    else:
        accuracy = 0.0

    if accuracy == 1.0:
        partial_accuracy = 1.0
        error_report = None
    else:
        ground_truth_procedure = example["reference_output"]
        partial_accuracy, error_report = evaluate_travel_plan_search_procedure(data_item, prediction, ground_truth_procedure)
    return {
        "accuracy": accuracy,
        "partial_accuracy": partial_accuracy,
        "extraction_rate": extraction_rate,
    }, {"parsed_output": plan_text, "error_report": error_report}


def _load_travel_planning_data(dataset_name: str, path: str=None) -> Tuple[Dict, Callable]:
    assert dataset_name in ["travel_planning_2k", "travel_planning_8k"]

    if path is None: path = "longproc_data"
    path = os.path.join(path, "travel_planning")

    data_file = os.path.join(path, "travel_planning_all.json")
    with open(data_file, "r") as f:
        data = json.load(f)

    with open(os.path.join(path, "prompts.yaml"), "r") as f:
        prompt = yaml.safe_load(f)
        user_prompt = prompt['USER_PROMPT']

    if "_2k" in dataset_name:
        output_range = (0, 2048)
    elif "_8k" in dataset_name:
        output_range = (4096, 8192)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    data = [d for d in data if output_range[0] <= d["estimated_output_tokens"] < output_range[1]]
    with open(os.path.join(path, "prompts.yaml"), "r") as f:
        prompt = yaml.safe_load(f)
        user_prompt = prompt['USER_PROMPT']

    def build_icl_demonstration():
        _DEMO_TEMPLATE = "# Example\n{problem}\n\n{solution}"
        with open(os.path.join(path, "travel_planning_icl_examples.json")) as f:
            data = json.load(f)
        demo = []
        for d in data:
            problem = d["disambig_question_text"]
            solution = build_travel_plan_demonstration(d)
            demo.append(_DEMO_TEMPLATE.format(problem=problem, solution=solution))
        return "\n\n".join(demo)

    # partially fill the user prompt
    user_prompt = user_prompt.format(demonstration=build_icl_demonstration(), problem="{problem}")

    data_purged = []
    for d in data:
        ground_truth_procedure = build_travel_plan_demonstration(d)
        disambig_question_text = d["disambig_question_text"]

        data_purged.append({
            "id": d["id"],
            "problem": disambig_question_text,
            "ground_truth_plan": d["ground_truth_plan"],
            "ground_truth_cities": d["ground_truth_cities"],
            "ground_truth_durations": d["ground_truth_durations"],
            "reference_output": ground_truth_procedure,
        })

    return {
        "data": data_purged,
        "prompt_template": user_prompt,
    }, eval_travel_planning


def load_longproc_data(dataset_name: str, path: str=None) -> Tuple[List, Callable]:
    """
    Load the dataset and evaluation function given the dataset name and path.
    returns: list of data, evaluation function
    the data list will contain {"input_prompt", "reference_output", and "item"} for each data point
    """

    dataset_basename = dataset_name.rsplit("_", 1)[0]

    dataset_loaders = {
        "html_to_tsv": _load_html_to_tsv_data,
        "pseudo_to_code": _load_pseudo_to_code_data,
        "path_traversal": _load_path_traversal_data,
        "tom_tracking": _load_tom_tracking_data,
        "countdown": _load_countdown_data,
        "travel_planning": _load_travel_planning_data,
    }
    
    if dataset_basename in dataset_loaders:
        dataset_loading_func = dataset_loaders[dataset_basename]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    packed_data, eval_func = dataset_loading_func(dataset_name, path)

    template = packed_data["prompt_template"]
    data = packed_data["data"]

    upacked_data = []
    for d in data:
        upacked_data.append({
            "input_prompt": template.format(**d),
            "reference_output": d["reference_output"],
            "item": d
        })

    assert all(["input_prompt" in d and "reference_output" in d and "item" in d for d in upacked_data])

    return upacked_data, eval_func

