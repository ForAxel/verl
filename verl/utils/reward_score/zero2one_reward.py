import re
import random
import signal
from typing import Optional
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from math_verify.errors import TimeoutException

_verify = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)

def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]



# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()

def is_correct_minerva(solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)<think>.*</think>(.*)") -> tuple[bool, str]:
    # Extract answer from solution
    match = re.findall(answer_pattern, solution_str, re.DOTALL)
    if not match:
        return False, "[INVALID]"
    pred = normalize_final_answer(remove_boxed(last_boxed_only_string(match[-1])))

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred

def contains_chinese(text: str) -> bool:
    # 检测中文字符
    has_chinese = bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf]', text))
    return has_chinese

def contains_repeat(text: str, n: int = 3) -> bool:
    # 检测重复
    tokens = re.findall(r'\b\w+\b', text)
    seen = set()
    for i in range(len(tokens) - n + 1):
        if (gram := tuple(tokens[i:i+n])) in seen:
            return True
        seen.add(gram)
    return False

def format_reward(solution_str: str) -> float:
    pattern = re.compile(r"(?i)<think>.*?</think>\s*<answer>.*?\\boxed\{(.*?)\}.*?</answer>", re.DOTALL) # 非贪婪匹配，只匹配第一对
    match_result = re.search(pattern, solution_str)
    if match_result:
        # return 0.1
        return 0 #qwen3-instruct format不work，直接取消format奖励
    return 0

def is_correct_int(solution_str, ground_truth):
    match = re.findall(r"\\boxed\{(.*?)\}", solution_str, re.DOTALL)
    if not match:
        return False, "[INVALID]"

    pred_str = match[-1]

    try:
        if float(pred_str) == float(ground_truth):
            return True, pred_str
        return False, pred_str
    except:
        return False, pred_str

def is_correct_by_math_verify(solution_str, ground_truth):
    # 调用第三方math_verify工具来判断两个答案是否相同，支持浮点数、根号、分数等

    match = re.search(r"(?i)<think>.*?</think>(.*)", solution_str, re.DOTALL)
    if not match:
        return False, "[INVALID]"

    answer_content = match.group(1).strip()
    string_in_last_boxed = last_boxed_only_string(answer_content) # 提取最后一个boxed{}内的内容
    if string_in_last_boxed is None:
        return False, "[INVALID]"

    pred_str = remove_boxed(string_in_last_boxed)
    try:
        ground_truth_boxed = "\\boxed{" + ground_truth + "}"
        pred_boxed = "\\boxed{" + pred_str + "}"
        ret_score, _ = _verify([ground_truth_boxed], [pred_boxed])
        return ret_score == 1.0, pred_str
    except:
        return False, pred_str

# def is_correct_int(solution_str, ground_truth):
#     match = re.search(r"(?i)<think>.*?</think>\s*<answer>.*?\\boxed\{(.*?)\}.*?</answer>", solution_str, re.DOTALL)  # 非贪婪匹配
#     if not match:
#         return False, "[INVALID]"

#     pred_str = match.group(1).strip()

#     try:
#         if float(pred_str) == float(ground_truth):
#             return True, pred_str
#         return False, pred_str
#     except:
#         return False, pred_str

# def format_reward(solution_str: str) -> float:
#     # 匹配 <think>...</think> 后至少有一个非空白字符（不一定要紧挨着）
#     pattern = re.compile(r"(?i)<think>.*?</think>.*?[^\s]", re.DOTALL)
#     match_result = re.search(pattern, solution_str)
#     if match_result:
#         return 0.1
#     return 0

# def is_correct_int_flw(solution_str, ground_truth):
#     # 保证和富老师评测脚本一致
#     match = re.search(r"(?i)<think>.*</think>.*\\boxed\{([^}]*)\}", solution_str, re.DOTALL)
#     if not match:
#         return False, "[INVALID]"

#     pred_str = match.group(1).strip()

#     try:
#         if str(pred_str).strip() == str(ground_truth).strip():
#             return True, pred_str
#         return False, pred_str
#     except:
#         return False, pred_str
import os
def compute_score(solution_str, ground_truth, data_source, extra_info=None, prompt_str=None, format_score=0., score=1., is_train_data=True, step=None):
    # do_print = random.randint(1, 16) == 1
    do_print = True
    # do_print = False
    # if random.random()<0.1:
    #     do_print = True
    
    solution_str = solution_str.strip()
    is_correct, pred = is_correct_int(solution_str, ground_truth) # 仅支持整数答案
    # is_correct, pred = is_correct_minerva(solution_str, ground_truth) # 参考math_dapo.py，支持非整数答案

    if do_print==True:
        printfile=os.getenv("printfile","printscore_curr")
        log_dir = os.getenv("REWARD_LOG_DIR", "/mnt/seed17/001688/shenyichong/verl/reward_logs")
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception as e:
                print(f"Error creating log directory {log_dir}: {e}")
        
        log_path = os.path.join(log_dir, f"{printfile}.txt")
        try:
            score_write=open(log_path,'a')
        except Exception as e:
            print(f"Error opening log file {log_path}: {e}")
            score_write = None
    if do_print:
        # print('--------------------------------')
        # print(f'Using 021 reward!')
        
        score_write.write('--------------------------------')
        score_write.write(f'Using 021 reward!')
        if step is not None:
            score_write.write(f'###step: {step} ###')
    solution_str = solution_str.strip()
    is_correct, pred = is_correct_int(solution_str, ground_truth) # 仅支持整数答案
    # is_correct, pred = is_correct_minerva(solution_str, ground_truth) # 参考math_dapo.py，支持非整数答案

    if is_train_data:
        if do_print:
            score_write.write(f"Compute score for train data")
        format_score = format_reward(solution_str)
    else:
        if do_print:
            score_write.write(f"Compute score for validation data\n")

    if do_print:
        score_write.write(f"Data source: {data_source}")
        score_write.write(f"Prompt:\n{prompt_str}")
        score_write.write("\n")
        score_write.write(f"Response:\n{solution_str}")
        score_write.write("\n")
        score_write.write(f"Ground truth: {ground_truth}")
        score_write.write("\n")
        score_write.write(f"Extracted answer: {pred}")
        score_write.write("\n")

    # 奖励计算
    if not is_correct:
        # 回答错误
        if is_train_data:
            # 训练数据有格式奖励
            total_score = format_score
        else:
            # 验证数据没有格式奖励
            total_score = 0.0
        if pred == "[INVALID]":
            if do_print:
                score_write.write(f"[ERROR] No answer extracted, format score: {format_score}, acc score: 0, total score: {total_score}")
        else:
            if do_print:
                score_write.write(f"[INCORRECT] Answer mismatch (extracted: {pred} ≠ expected: {ground_truth}), format score: {format_score}, acc score: 0, total score: {total_score}")

        return {
            "score": total_score,
            "format_score": format_score,
            "accuracy_score": 0.0
        }
    else:
        # 回答正确
        if is_train_data:
            # 训练数据有格式奖励
            total_score = score + format_score
        else:
            # 验证数据没有格式奖励
            total_score = score
        if do_print:
            score_write.write(f"[CORRECT] Answer matches ground truth, format score: {format_score}, acc score: {score}, total score: {total_score}")

        return {
            "score": total_score,
            "format_score": format_score,
            "accuracy_score": score
        }
