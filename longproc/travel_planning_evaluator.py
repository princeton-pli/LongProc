import json
import re
import string

_ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_INIT_CITY = "the starting point"

class _LogManager:
    def __init__(self) -> None:
        self.log = []

    def append(self, msg):
        self.log.append(msg)

def get_indent(level: int):
    if level == 0:
        return ""
    else:
        return " " * level + "|- "

def print_list(lst):
    return f'[{", ".join([str(x) for x in lst])}]'

def _verbalized_dfs_search(current_day, num_cities, current_schedule,
                           free_cities, fixed_cities, all_connections,
                           log_manager: _LogManager) -> bool:
    indent_depth = len(current_schedule) * 2
    log_manager.append(f"{get_indent(indent_depth)}Current day: {current_day}. Current plan: {print_list([x['city'] for x in current_schedule])}.")

    if len(current_schedule) == num_cities:
        log_manager.append(f"{get_indent(indent_depth)}All {num_cities} cities are arranged. Complete plan is found!")
        return current_schedule
    # day gather options for this 
    log_manager.append(f"{get_indent(indent_depth)}Check whether the city with an arrival day of Day {current_day} - is fixed.")
    fixed_option = next((city for city in fixed_cities if city["start_day"] == current_day), None)
    if fixed_option is not None:
        log_manager.append(f"{get_indent(indent_depth)}Yes. The city with an arrival day of Day {current_day} - is fixed: {fixed_option['city']}.")
        choices = [{
            "city": fixed_option["city"],
            "num_days":fixed_option["end_day"] - fixed_option["start_day"] + 1,
            "start_day": fixed_option["start_day"],
            "end_day": fixed_option["end_day"]
        }]
    else:
        log_manager.append(f"{get_indent(indent_depth)}No. Consider possible options from cities needing arrangement: {print_list([x['city'] for x in free_cities])} and explore these options in order.")
        choices = []
        for free_city in free_cities:
            choice = {
                "city": free_city["city"],
                "num_days": free_city["num_days"],
                "start_day": current_day,
                "end_day": current_day + free_city["num_days"] - 1,
            }
            choices.append(choice)

    # check possiblity of placement
    # following city on schedule
    last_city = _INIT_CITY if len(current_schedule) == 0 else current_schedule[-1]["city"]
    following_fixed =  next((city for city in fixed_cities if city["start_day"] > current_day), None)

    indent_depth += 1
    for choice in choices:
        choice_city, choice_start_day, choice_end_day = choice["city"], choice["start_day"], choice["end_day"]
        log_manager.append(f"{get_indent(indent_depth)}Try arranging to visit {choice['city']} from Day {current_day}. Duration: {choice['num_days']} days. Schedule: Day {current_day} - {current_day + choice['num_days'] - 1}.")
        # check compatibility with connect
        log_manager.append(f"{get_indent(indent_depth)}Check for direct flight from {last_city} to {choice_city}.")
        if not (last_city == _INIT_CITY or (last_city, choice_city) in all_connections):
            log_manager.append(f"{get_indent(indent_depth)}No. Drop this branch.")
            continue
        else:
            log_manager.append(f"{get_indent(indent_depth)}Yes.")

        if following_fixed is None:
            log_manager.append(f"{get_indent(indent_depth)}Check whether this arrangement is compatible with the next fixed schedule after Day {current_day}: None.")
        else:
            log_manager.append(f"{get_indent(indent_depth)}Check whether this arrangement is compatible with the next fixed schedule after Day {current_day}: {following_fixed['city']} (Day {following_fixed['start_day']} - {following_fixed['end_day']}).")
    
        # check compatibility with schedule
        if following_fixed is None:
            log_manager.append(f"{get_indent(indent_depth)}No following fixed schedules. This arrangement is compatible.")
        elif following_fixed["start_day"] == choice_start_day:
            raise RuntimeError("Invalid fixed schedule.")
        elif following_fixed["start_day"] < choice_end_day:
            log_manager.append(f"{get_indent(indent_depth)}The departure day of {choice_city} is Day {choice_end_day}. The arrival day of {following_fixed['city']} is Day {following_fixed['start_day']}. Day {choice_end_day} is later than (>) Day {following_fixed['start_day']}. This arrangement is incompatible. Drop this branch.")
            continue
        else:
            log_manager.append(f"{get_indent(indent_depth)}The departure day of {choice_city} is Day {choice_end_day}. The arrival day of {following_fixed['city']} is Day {following_fixed['start_day']}. Day {choice_end_day} is not later than (<=) Day {following_fixed['start_day']}. This arrangement is compatible.")
        log_manager.append(f"{get_indent(indent_depth)}This arrangement is feasible for now. Continue to arrange the rest of the plan.")
        branch_result = _verbalized_dfs_search(
            choice_end_day, num_cities, current_schedule + [choice], 
            [x for x in free_cities if x["city"] != choice_city],
            fixed_cities,
            all_connections,
            log_manager
        )
        if branch_result is not None:
            return branch_result

    log_manager.append(f"{get_indent(indent_depth)}Fail to arrange any option on day {current_day} in the current arrangement. Drop this branch.")
    return None


def format_result_plan(schedule: list):
    lines = []
    for i, stop in enumerate(schedule):
        city, start_day, end_day, num_days = stop["city"], stop["start_day"], stop["end_day"], stop["num_days"]
        if i == 0:
            lines.append(f"**Day {start_day}-{end_day}:** Arriving in {city} and visit {city} for {num_days} days.")
        else:
            lines.append(f"**Day {start_day}-{end_day}:** Visit {city} for {num_days} days.")
        if i + 1 < len(schedule):
            next_city = schedule[i + 1]["city"]
            lines.append(f"**Day {end_day}:** Fly from {city} to {next_city}.")
    return "\n".join(lines)


def build_travel_plan_demonstration(ex):
    num_cities = ex["num_cities"]
    total_days = ex["total_days"]
    constraints = ex["constraints"]
    connected_cities = ex["connected_cities"]

    connected_cities = [tuple(x) for x in connected_cities]
    # seperate fix schedule constraints and duration constraints
    fixed_constraints = [cons for cons in constraints if cons["type"] == "fixed"]
    duration_constraints = [cons for cons in constraints if cons["type"] == "duration"]
    fixed_cities = [cons["city"] for cons in fixed_constraints]
    free_cities = [cons for cons in duration_constraints if cons["city"] not in fixed_cities]
    fixed_cities = fixed_constraints

    log_manager = _LogManager()

    log_manager.append(f"Read the requirements and identify the cities that have fixed schedules and the cities that need to be arranged.")

    for i, cons in enumerate(constraints):
        if cons["type"] == "fixed":
            continue
        if any([cons["city"] == x["city"] for x in fixed_cities]):
            assert constraints[i + 1]["type"] == "fixed" and constraints[i + 1]["city"] == cons["city"]
            fixed_cons = constraints[i + 1]
            log_manager.append(f"* City: {cons['city']}, Duration: {cons['num_days']} days, Fixed Schedule: Day {fixed_cons['start_day']} - {fixed_cons['end_day']}.")
        else:
            log_manager.append(f"* City: {cons['city']}, Duration: {cons['num_days']} days.")
    log_manager.append("")

    if fixed_cities:
        log_manager.append(f"Cities that have fixed schedules (sorted by their arrival days):")
        fixed_cities = sorted(fixed_cities, key=lambda x: x["start_day"])
        for fixed_city in fixed_cities:
            log_manager.append(f"* City: {fixed_city['city']}, Fixed Schedule: Day {fixed_city['start_day']} - {fixed_city['end_day']}.")
    else:
        log_manager.append(f"Cities that have fixed schedules: None.")
    log_manager.append("")

    if free_cities:
        log_manager.append(f"Cities needing arrangement:")
        for free_city in free_cities:
            log_manager.append(f"* City: {free_city['city']}, Duration: {free_city['num_days']} days.")
    else:
        log_manager.append(f"Cities needing arrangement: None.")
    log_manager.append("")

    schedule = _verbalized_dfs_search(1, num_cities, [], free_cities, fixed_cities, connected_cities, log_manager)
    predicted_plan = format_result_plan(schedule)
    assert predicted_plan == ex["ground_truth_plan"].split("\n\n")[1].strip()

    search_log = "\n".join(log_manager.log)
    procedure = f"<Solving Procedure>\n{search_log}\n</Solving Procedure>\n\nOutput the plan in the required format:\n<Plan>\n{predicted_plan}\n</Plan>"
    return procedure

# code from natural plan benchmark
def _parse_response(response: str):
  """Parse the response.

  Returns a parsed plan in a list of (city, stay_days) tuples.

  Args:
    response: Raw response from the model.

  Returns:
    Structured plan after parsing.
  """
  pattern_visit = r'\d+-\d+'
  pattern_flight = r'.*Day (\d+).*from (\w+) to (\w+)'
  pattern_days = r'European cities for (\d+) days'

  days, flights, flight_days = [], [], []
  total_days = None
  for piece in response.split('\n'):
    days_match = re.findall(pattern_days, piece)
    if days_match:
      total_days = int(days_match[0])

    visit_match = re.findall(pattern_visit, piece)
    if visit_match:
      days.append(visit_match[0])
      end_day = int(visit_match[0].split('-')[1])
      # Reach the end of the plan, stop to avoid parsing alternative plans.
      if end_day == total_days:
        break
    flight_match = re.findall(pattern_flight, piece)
    if flight_match:
      flights.append(flight_match[0])

  visit_cities, parsed_plan = [], []
  for flight_day, begin_city, end_city in flights:
    flight_days.append(int(flight_day))
    if not visit_cities:
      visit_cities.append(begin_city)
      visit_cities.append(end_city)
    else:
      visit_cities.append(end_city)

  if not days or not flights or not visit_cities:
    return []
  last_day = int(days[-1].split('-')[1])
  flight_days = [1] + flight_days + [last_day]
  for i, visit_city in enumerate(visit_cities):
    city_stay = flight_days[i + 1] - flight_days[i] + 1
    parsed_plan.append((visit_city, city_stay))

  return parsed_plan


def evaluate_travel_plan_solution(cities: str, durations: str, response: str):
  """Compute the exact-match accuracy.

  Compute the example-level exact_match score (0/1) given the parsed plan
  and the ground truth in the format of durations and cities.

  Args:
    cities: The cities in the plan in the format of "city1**city2**city3".
    durations: The durations of the stay in each city in the format of
      "1**2**3".
    parsed_plan: The parsed plan from the response.

  Returns:
    Exact-match accuracy of 0 (mismatched) or 1 (matched).
  """

  stays = [x for x in cities.split('**') if x]
  days = [int(x) for x in durations.split('**') if x]
  parsed_plan = _parse_response(response)
  num_stays = min(len(stays), len(parsed_plan))
  num_match = 0
  for i in range(num_stays):
    if stays[i] == parsed_plan[i][0] and days[i] == parsed_plan[i][1]:
      num_match += 1
    else:
      break
  hard_score = 0.0 if num_match / len(stays) < 1.0 else 1.0
  return hard_score


def _normalize_line(line: str):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return _ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(line))))

def evaluate_travel_plan_search_procedure(example: dict, output: str, gt_procedure: str) -> float:
    # remove some unnecessary information
    gt_procedure = gt_procedure.replace("Output the plan in the required format:", "")
    output = output.replace("Output the plan in the required format:", "")
    gt_procedure =  "<Solving Procedure>" + gt_procedure.split("<Solving Procedure>")[1]
    if "<Solving Procedure>" in output:
        output = "<Solving Procedure>" + output.split("<Solving Procedure>")[1]

    pred_lines = output.strip().split("\n")
    gt_lines = gt_procedure.strip().split("\n")

    pred_lines = [line.rstrip() for line in pred_lines if line.strip()]
    gt_lines = [line.rstrip() for line in gt_lines if line.strip()]

    idx = -1
    error_report = {}
    for pred_l, gt_l in zip(pred_lines, gt_lines):
        idx += 1
        # fast forward with the same lines
        _pred_l = _normalize_line(pred_l)
        _gt_l = _normalize_line(gt_l)
        if _gt_l in _pred_l:
            # print(gt_l)
            continue
        else:
            error_report = {"line": idx, "gt": pred_l, "pr": gt_l}
            break
    if idx < 0:
        idx = 0
        error_report = {"empty_output": True}

    return idx / len(gt_lines), error_report
