import re
from typing import Union, Optional, List
from model import generator

def extract_number(args, text: str) -> Union[float, None]:
    text = text.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', text)]
    if pred:
        pred_answer = float(pred[-1])
    else:
        pred_answer = None
    return pred_answer

def answer_cleanser(args: any, text: str) -> Union[str, float]:
    dataset = args.dataset.lower()
    if dataset in ["svamp", "gsm8k", "multiarith", "addsub", "singleeq"]:
        pred_answer = extract_number(args, text)
    elif dataset == "commonsenseqa":
        pred = text.strip()
        pred = re.sub("\(|\)|\:|\.|\,", "", pred)
        pred = pred.split()
        pred_answer = [i for i in pred if i in ('A|B|C|D|E')][-1]
        # pred_answer = re.findall(r'A|B|C|D|E', pred)[0]
        return pred_answer
    elif dataset == "aqua":
        pred = text.strip()
        pred_answer = re.findall(r'A|B|C|D|E', pred)[0]
        return pred_answer
    elif dataset == "strategyqa" or dataset == 'coin_flip':
        pred = text.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split()
        pred_answer = [i for i in pred if i in ("yes", "no")][-1]
        return pred_answer
    elif dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s", "", text)
        pred_answer = pred
        return pred_answer
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer(args: any, prompt: str, predictions: List[str], generator: generator) -> Union[str, float]:
    prompts = []
    for prediction in predictions:
        prompts.append(prompt + prediction + "Therefore, the answer is ")
    outputs = generator.generate(prompt=prompts,
                                 max_length=32,
                                 temperature=0)

    prompt2pred = {}
    for output in outputs:
        prompt2pred[output.prompt] = output.outputs

    predictions_ = []
    full_predictions = []
    for prompt, prediction in zip(prompts, predictions):
        output = prompt2pred[prompt][0]
        #print("prediction: ", prediction)
        #print("answer prompt result: ", output.text)
        #print("\n\n\n")

        if 'Therefore, the answer is' in prediction or 'The answer is' in prediction:
            if 'The answer is' in prediction:
                pred2 = prediction.split('The answer is')[-1]
            else:
                pred2 = prediction.split('the answer is')[-1]
            full_predictions.append(prediction)
        else:
            pred2 = output.text
            full_predictions.append(prediction + "Therefore, the answer is " + pred2)

        try:
            pred2 = answer_cleanser(args, pred2)
        except:
            pred2 = None
        predictions_.append(pred2)

    return predictions_, full_predictions
