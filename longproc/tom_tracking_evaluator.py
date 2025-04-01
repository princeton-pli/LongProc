import string
import re

def _normalize_tom(s):
    """
    Lowercase, remove punctuation, articles, and extra whitespace.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the|on|in|at|the|step|thinks|think|believes|believe|is|are|of|location|know|knows|belief)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punctuation(text):
        return ''.join(ch for ch in text if ch not in string.punctuation and ch!='’')
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punctuation(lower(s))))


def _extract_belief_content(line):
    # line should start with - Step and a numer
    if re.match(r'^- Step \d+:', line):
        # Split on the first hyphen and get the content after it
        belief_content = line.split('-', 1)[1].strip()
        # Normalize the belief content
        belief_content = _normalize_tom(belief_content)
        return belief_content
    else:
        return None

def evaluate_tom_trace(model_response_text, ground_truth_text):
    # start eval from the last place where the trace starts, ignore internal thinking inside cot
    if "- Step 0:" in model_response_text:
        start_index = model_response_text.rfind("- Step 0:")
        model_response_text = model_response_text[start_index:]
    model_responses = model_response_text.strip().split('\n')
    ground_truth_responses = ground_truth_text.strip().split('\n')

    # Process the lines to extract belief contents
    model_beliefs = [_extract_belief_content(line) for line in model_responses if _extract_belief_content(line)]
    ground_truth_beliefs = [_extract_belief_content(line) for line in ground_truth_responses if _extract_belief_content(line)]

    # strict accuracy
    error_report = None
    if len(model_beliefs) == len(ground_truth_beliefs) and all(a == b for a, b in zip(model_beliefs, ground_truth_beliefs)):
        strict_accuracy = 1.0
        partial_accuracy = 1.0
    else:
        strict_accuracy = 0.0
        first_diff = next((i for i, (a, b) in enumerate(zip(model_beliefs, ground_truth_beliefs)) if a != b), None)
        if first_diff is not None:
            partial_accuracy = first_diff / len(ground_truth_beliefs)
            error_report = {
                "line": first_diff,
                "pr": model_beliefs[first_diff],
                "gt": ground_truth_beliefs[first_diff]
            }
        else:
            # Handle case where there are no mismatches but lengths are different
            partial_accuracy = min(len(model_beliefs), len(ground_truth_beliefs)) / len(ground_truth_beliefs)

    return strict_accuracy, partial_accuracy, error_report
