from typing import Union

def get_precision(gt_ans: float) -> int:
    precision = 5
    if '.' in str(gt_ans):
        precision = len(str(gt_ans).split('.')[-1])
    return precision

def evaluate(args, pred_answer: Union[None, str, float], answer: Union[str, float]) -> int:
    epsilon = 1e-3
    correct = 0
    if pred_answer is None:
        return 0
    if args.dataset.lower() in ["svamp", "gsm8k", "multiarith", "addsub", "singleeq"]:
        if abs(pred_answer - answer) < epsilon:
            correct = 1
    else:
        if isinstance(pred_answer, float) and isinstance(answer, float):
            precision = min(get_precision(pred_answer), get_precision(answer))
            if round(pred_answer, precision) == round(answer, precision):
                correct = 1
        else:
            if pred_answer == answer:
                correct = 1

    return correct