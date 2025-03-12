from typing import (
    List,
    Any
)

from sqlitedict import SqliteDict
from llm_utils import batch_query, key_of_llm_query

# NOTE: sampling logits are implemented here
def cached_batch_query(
    cache_path: str,
    prompts: List[Any],
    model_name: str,
    gen_max_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    max_model_len: int = None,
    n: int = 1,
    query_kwargs: dict = {},
    aux_kwargs: dict = {},
    ):
    assert n == 1 or temperature > 0.0
    force_overwrite = aux_kwargs.get("force_overwrite", False)

    cache = SqliteDict(cache_path, autocommit=True)

    all_responses = []
    todo_queries = []

    for prompt_id, prompt in enumerate(prompts):
        prompt_response = [None] * n
        for sample_id in range(n):
            # aux kwargs are not part of the key
            key = key_of_llm_query(prompt, model_name, gen_max_tokens, temperature, top_p, max_model_len, **query_kwargs, sample_id=sample_id)
            if key in cache and not force_overwrite:
                # print(f"Cache hit for {prompt} {model_name}")
                prompt_response[sample_id] = cache[key]
                assert prompt_response[sample_id]["prompt"] == prompt
            else:
                todo_queries.append(((prompt_id, sample_id), key, prompt))
        all_responses.append(prompt_response)

    # query
    if len(todo_queries):
        # avoid starting the model if there are no queries
        todo_responses = batch_query([x[2] for x in todo_queries], model_name, gen_max_tokens, temperature, top_p, max_model_len, query_kwargs, aux_kwargs)
    else:
        todo_responses = []

    for ((prompt_id, sample_id), key, prompt), resp in zip(todo_queries, todo_responses):
        cache[key] = resp
        assert all_responses[prompt_id][sample_id] is None
        all_responses[prompt_id][sample_id] = resp

    cache.close()
    assert all(all(x is not None for x in prompt_response) and len(prompt_response) == n for prompt_response in all_responses)

    # merge responses
    merged_responses = []
    for prompt_response in all_responses:
        output = prompt_response[0]
        output["output"] = [resp["output"] for resp in prompt_response]
        merged_responses.append(output)
    return merged_responses


def __test_OAI_batch_query():
    import random
    random.seed(0)
    prompts = []
    for _ in range(3):
        prompts.append([{"role": "user", "content": f"Count from 1 to {random.randint(1, 50)}"}])
    
    outputs = cached_batch_query("caches/gpt4o-mini.sqlite", prompts, "gpt-4o-mini", 100, 0.5, aux_kwargs={"force_overwrite": True})
    for prompt, output in zip(prompts, outputs):
        print(prompt)
        print(output)
        print()

    more_prompts = []
    for _ in range(3):
        more_prompts.append([{"role": "user", "content": f"Count from 5 to {random.randint(5, 50)}"}])

    more_prompts = prompts + more_prompts
    random.shuffle(more_prompts)
    more_outputs = cached_batch_query("caches/gpt4o-mini.sqlite", more_prompts, "gpt-4o-mini", 100, 0.5, aux_kwargs={"force_overwrite": False})
    for prompt, output in zip(more_prompts, more_outputs):
        print(prompt)
        print(output)
        print()


def __test_vllm_batch_query():
    import random
    random.seed(0)
    model_name = "vllm/models/Llama-3.2-3B-Instruct"
    cache_name = "caches/" + model_name.replace("/", "-") + ".sqlite"

    prompts = []
    for _ in range(3):
        prompts.append([{"role": "user", "content": f"Count from 1 to {random.randint(1, 50)}"}])
    
    outputs = cached_batch_query(cache_name, prompts, model_name, 100, 0.5, max_model_len=8192, aux_kwargs={"force_overwrite": True})
    for prompt, output in zip(prompts, outputs):
        print(prompt)
        print(output)
        print()

    more_prompts = []
    for _ in range(3):
        more_prompts.append([{"role": "user", "content": f"Count from 5 to {random.randint(5, 50)}"}])

    more_prompts = prompts + more_prompts
    random.shuffle(more_prompts)
    more_outputs = cached_batch_query(cache_name, more_prompts, model_name, 100, 0.5, max_model_len=8192, aux_kwargs={"force_overwrite": False})
    for prompt, output in zip(more_prompts, more_outputs):
        print(prompt)
        print(output)
        print()


def __test_OAI_batch_query_sampling():
    import random
    random.seed(0)
    prompts = []
    for _ in range(3):
        prompts.append([{"role": "user", "content": f"Count from 1 to {random.randint(1, 50)}"}])
    
    outputs = cached_batch_query("caches/gpt4o-mini.sqlite", prompts, "gpt-4o-mini", 100, 1.0, 0.95, n=4)
    for prompt, output in zip(prompts, outputs):
        print(prompt)
        print(output["output"])
        print()

def __test_vllm_batch_query_sampling():
    import random
    random.seed(0)
    model_name = "vllm/models/Llama-3.2-3B-Instruct"
    cache_name = "caches/" + model_name.replace("/", "-") + ".sqlite"

    prompts = []
    for _ in range(3):
        prompts.append([{"role": "user", "content": f"Count from 1 to {random.randint(1, 50)}"}])
    
    outputs = cached_batch_query(cache_name, prompts, model_name, 100, 1.0, 0.95, max_model_len=8192, n=6)
    for prompt, output in zip(prompts, outputs):
        print(prompt)
        print(output["output"])
        print()


if __name__=="__main__":
    __test_OAI_batch_query_sampling()
    __test_vllm_batch_query_sampling()
    # __test_OAI_batch_query()
    # __test_vllm_batch_query()
