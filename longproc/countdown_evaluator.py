### This file is used to
### 1) generate the search procedure for the countdown problem
### 2) eavluate the final solution as well as the search procedure
import json


_INIT_TRANSITION = "INIT_TRANS"
_MAX_INTERMEDIATE_RESULTS = 2000
_DIV_BY_ZERO = "DIV_BY_ZERO"

_DUMMY_STATE_ID = -1

_FORMAT_SOLUTION_TEMPLATE = """
Now we have found the target, let's trace back the solution:
Final step: {solution[2]}
The step before: {solution[1]}
The first step: {solution[0]}

Output the solution in the required format:
<Solution>
{solution[0]}
{solution[1]}
{solution[2]}
</Solution>
""".lstrip()


def _get_indent(level: int):
    if level == 0:
        return ""
    else:
        return " " * level + "|- "


def _choose_two_numbers(nums: list[int]):
    for i, a in enumerate(nums):
        for j, b in enumerate(nums):
            if j <= i:
                continue
            yield a, b, [x for k, x in enumerate(nums) if k != i and k != j]


def _verbalized_dfs_search(
        nums: list[int],
        target: int,
        parent_state_id: int,
        transition_eq: str,
        recursion_depth: int,
        explored_states: list,
        search_log: list,
        indent_func,
    ):
    state_id = len(explored_states)
    state = {
        'state_id': state_id,
        'parent_state_id': parent_state_id,
        'transition_eq': transition_eq,
        'nums': nums,
        'target': target,
        'depth': recursion_depth,
    }
    explored_states.append(state)
    if not transition_eq == _INIT_TRANSITION:
        if nums[0] == _DIV_BY_ZERO:
           trans_eval_log = f"\n{indent_func(recursion_depth)}Try {transition_eq}. drop this branch."
           search_log.append(trans_eval_log)
           return False 
        assert nums[0] >= 0
        # Transition evaluation, if nums[0] is float or negative, return False
        if nums[0] % 1 != 0:
            trans_eval_log = f"\n{indent_func(recursion_depth)}Try {transition_eq}. {nums[0]:.1f} is a decimal, drop this branch." 
            search_log.append(trans_eval_log)
            return False
        elif nums[0] >= _MAX_INTERMEDIATE_RESULTS:
            trans_eval_log = f"\n{indent_func(recursion_depth)}Try {transition_eq}. {nums[0]} exceeds the maximum intermediate result, drop this branch."
            search_log.append(trans_eval_log)
            return False
        else:
            pass

    # termination condition
    if len(nums) == 1:
        if nums[0] == target:
            trans_eval_log = f"\n{indent_func(recursion_depth)}Try {transition_eq}. Evaluate {nums[0]} == {target}, target found!"
            search_log.append(trans_eval_log)
            return True
        else:
            trans_eval_log = f"\n{indent_func(recursion_depth)}Try {transition_eq}. Evaluate {nums[0]} != {target}, drop this branch."
            search_log.append(trans_eval_log)
            return False

    # log the current state
    if not transition_eq == _INIT_TRANSITION:
        trans_eval_log = f"\n{indent_func(recursion_depth)}Try {transition_eq}. Add {nums[0]} to the number set."
        search_log.append(trans_eval_log)
    two_num_options = [(a,b) for a, b, _ in _choose_two_numbers(nums)]

    if len(nums) == 2:
        bigger_num = max(nums)
        smaller_num = min(nums)
        # state_descp_log = f"\n{indent_func(recursion_depth)}Current number set: {nums}, target: {target}, just two numbers left"
        state_descp_log = f" Current number set: {nums}, target: {target}, just two numbers left."
        search_log.append(state_descp_log)
    else:
        if not transition_eq == _INIT_TRANSITION:
            # state_descp_log = f"\n{indent_func(recursion_depth)}Current number set: {nums}, target: {target}, options for choosing two numbers: {two_num_options}"
            state_descp_log = f" Current number set: {nums}, target: {target}. Options for choosing two numbers: {two_num_options}."
        else:
            state_descp_log = f"Initial number set: {nums}, target: {target}. Options for choosing two numbers: {two_num_options}."
        search_log.append(state_descp_log)
        recursion_depth += 1

    # choose two numbers from nums
    for a, b, leftover in _choose_two_numbers(nums):
        bigger_num = max(a, b)
        smaller_num = min(a, b)
        if len(nums) > 2:
            pick_act_log = f"\n{indent_func(recursion_depth)}Pick two numbers ({a}, {b}) (numbers left: {leftover}). Try possible operations." 
            search_log.append(pick_act_log)
        # let a >= b
        if a < b:
            a, b = b, a
        # possible options: a + b, a - b, b - a, a * b, a / b, b / a
        # a + b
        if _verbalized_dfs_search([a + b] + leftover, target, state_id, f"{a} + {b} = {a+b}", recursion_depth + 1, explored_states, search_log, indent_func):
            return True
        # a - b
        if _verbalized_dfs_search([a - b] + leftover, target, state_id, f"{a} - {b} = {a-b}", recursion_depth + 1, explored_states, search_log, indent_func):
            return True
        # a * b
        if _verbalized_dfs_search([a * b] + leftover, target, state_id, f"{a} * {b} = {a*b}", recursion_depth + 1, explored_states, search_log, indent_func):
            return True
        # a / b
        if b != 0:
            if a % b == 0:
                if _verbalized_dfs_search([int(a / b)] + leftover, target, state_id, f"{a} / {b} = {int(a/b)}", recursion_depth + 1, explored_states, search_log, indent_func):
                    return True
            else:
                if _verbalized_dfs_search([a / b] + leftover, target, state_id, f"{a} / {b} = {a/b:.1f}", recursion_depth + 1, explored_states, search_log, indent_func):
                    return True
        else:
            if _verbalized_dfs_search([_DIV_BY_ZERO] + leftover, target, state_id, f"{a} / {b} (invalid operation)", recursion_depth + 1, explored_states, search_log, indent_func):
                return True
    #     search_log.append(f"\n{indent_func(recursion_depth + 1)}All possible operations tried for pair ({_in_a}, {_in_b}), backtrack")
    # search_log.append(f"\n{indent_func(recursion_depth)}All possible number pairs tried for {nums}, backtrack")
    return False


def build_countdown_demonstration(nums: list, target: int) -> str:

    search_states = []
    search_log = []
    success_status = _verbalized_dfs_search(nums, target, _DUMMY_STATE_ID, _INIT_TRANSITION, 0, search_states, search_log, _get_indent)
    if not success_status:
        raise ValueError("Failed to find solution")

    search_log = "".join(search_log)

    solution = []
    state = search_states[-1]
    while state['parent_state_id'] != _DUMMY_STATE_ID:
        solution.append(state['transition_eq'])
        state = search_states[state['parent_state_id']]
    solution.reverse()
    # verbolize the solution
    try:
        output_solution = _FORMAT_SOLUTION_TEMPLATE.format(solution=solution)
    except Exception as e:
        print(nums)
        print(target)
        print(search_states)
        print(search_log)
        print(solution)
        print(e)
        raise e

    demonstraion = "# Search Procedure\n" + search_log + "\n\n" + output_solution
    return solution, demonstraion


def evaluate_countdown_final_solution(nums: list, target: int, solution: str) -> bool:
    nums = nums[:]

    # parse a ? b = c into a, b, c, op
    def _parse_line(line):
        line = line.strip()
        if len(line.split("=")) != 2:
            return False, None, None, None, None
        lhs, rhs = line.split("=")
        lhs_result = eval(lhs)
        if "+" in lhs:
            op = "+"
        elif "-" in lhs:
            op = "-"
        elif "*" in lhs:
            op = "*"
        elif "/" in lhs:
            op = "/"
        else:
            return False, None, None, None, None,
        a, b = lhs.split(op)
        return lhs_result == int(rhs), int(a), int(b), int(rhs), op, 

    # parse solution into equations
    lines = solution.split("\n")
    if len(lines) != 3:
        return False
    # check if the solution is correct
    for line in lines:
        try:
            correct, a, b, c, op = _parse_line(line)
        except ValueError:
            return False
        except Exception as e:
            print(e)
            return False
        if not correct:
            return False
        if a not in nums:
            return False
        nums.remove(a)
        if b not in nums:
            return False
        nums.remove(b)
        nums.append(c)
    final_result = list(nums)[0]
    return final_result == target


def evaluate_countdown_search_procedure(nums: list, target: int, procedure: str, gt_procedure: str) -> tuple:
    # return partial accuracy as a float, and return error report
    # we focus on evaluating the "actions" in the procedure
    nums = nums[:]

    pred_lines = procedure.strip().split("\n")
    gt_lines = gt_procedure.strip().split("\n")

    # initalization statement should be the same
    if pred_lines[0] != gt_lines[0]:
        return 0.0, {"line_number": 0, "prediction": pred_lines[0], "ground_truth": gt_lines[0]}
    pred_lines = pred_lines[1:]
    gt_lines = gt_lines[1:]

    idx = -1
    error_report = {}
    for pred_l, gt_l in zip(pred_lines, gt_lines):
        idx += 1
        # fast forward with the same lines
        if pred_l == gt_l:
            continue
        # categorize the gt lines
        if "Pick two numbers" in gt_l: # pick numbers, it should follow the same order 
            if pred_l != gt_l:
                error_report = {"line": idx, "pr": pred_l, "gt": gt_l}
                break
        elif "|- Try" in gt_l: # try operation
            # everything up to the = should be the same, should be operating on the same numbers
            pred_eq = pred_l.split("=")[0]
            gt_eq = gt_l.split("=")[0]
            if pred_eq != gt_eq:
                error_report = {"line": idx, "pr": pred_l, "gt": gt_l}
                break
            # action should be the same
            dropping_in_gt = "drop this branch" in gt_l
            dropping_in_pred = "drop this branch" in pred_l
            if dropping_in_gt != dropping_in_pred:
                error_report = {"line": idx, "pr": pred_l, "gt": gt_l}
                break
            continue
        else:
            raise ValueError(f"Unknown line: {gt_l}")

    return idx / len(gt_lines), error_report


if __name__ == "__main__":
    # nums = [40, 19, 23, 7]
    # target = 29
    nums = [9, 16, 6, 18]
    target = 12

    demonstration = build_countdown_demonstration(nums, target)
    print(demonstration[1])