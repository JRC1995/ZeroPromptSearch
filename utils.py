import json
from typing import List

def argmax(iterable: List) -> int:
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def argsort(seq: List):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def load_data(args):
    decoder = json.JSONDecoder()
    questions = []
    answers = []
    ids = []
    datapath = 'Dataset/{}/{}.json'.format(args.dataset, args.dataset)
    if args.dataset == 'gsm8k_zct_8':
        questions, rational, answers = [], [], []
        datapath = 'result/ours/text003/gsm8k_zct_1_10_8.json'
        with open(datapath) as f:
            json_data = json.load(f)
            for idx, line in enumerate(json_data):
                q = line['question']
                r = line['chain-of-thought']
                a = line['answer']
                questions.append(q)
                rational.append(r)
                answers.append(a)
                ids.append('temp_{}'.format(idx))
        if args.test_num == 'full':
            return questions, rational, answers
        else:
            return questions[:int(args.test_num)], rational[:int(args.test_num)], answers[:int(args.test_num)]
    if args.dataset == 'CommonsenseQA':
        datapath = 'Dataset/CommonsenseQA/CommonsenseQA.jsonl'
    # read dataset file
    if args.dataset.lower() in ['svamp', 'svamp_sorted', 'gsm8k', 'gsm8k_sorted', 'multiarith', 'addsub', 'singleeq',
                                'strategyqa', 'coin_flip', 'last_letters']:
        with open(datapath) as f:
            if args.dataset.lower() in ['coin_flip', 'last_letters', 'strategyqa']:
                json_data = json.load(f)["examples"]
            else:
                json_data = json.load(f)

            for idx, line in enumerate(json_data):
                if args.dataset.lower() == 'svamp':
                    if line['Body'][-1] != '.':
                        q = line['Body'].strip() + ". " + line["Question"].strip()
                    else:
                        q = line['Body'].strip() + " " + line["Question"].strip()
                    a = float(line["Answer"])
                    id = line["ID"]
                elif args.dataset == 'svamp_sorted':
                    q = line['Question']
                    a = float(line['Answer'])
                    id = line['ID']
                elif args.dataset.lower() == 'strategyqa':
                    q = line["input"].strip()
                    a = int(line["target_scores"]["Yes"])
                    if a == 1:
                        a = "yes"
                    else:
                        a = "no"
                    id = 'temp_{}'.format(idx)
                elif args.dataset.lower() in ['coin_flip', 'last_letters']:
                    q = line["question"]
                    a = line["answer"]
                    id = 'temp_{}'.format(idx)
                elif args.dataset.lower() in ["multiarith", 'addsub', 'singleeq']:
                    q = line['sQuestion']
                    a = float(line['lSolutions'][0])
                    id = 'temp_{}'.format(idx)
                elif args.dataset.lower() in ['gsm8k', 'gsm8k_sorted', 'examples', 'examples']:
                    q = line['question']
                    a = float(line['answer'])
                    id = 'temp_{}'.format(idx)
                else:
                    raise ValueError('not support dataset: {}'.format(args.dataset))
                questions.append(q)
                answers.append(a)
                ids.append(id)

    elif args.dataset.lower() in ['aqua', 'commonsenseqa']:
        with open(datapath) as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if args.dataset.lower() == 'aqua':
                    json_res = decoder.raw_decode(line)[0]
                    choice = "(" + "(".join(json_res["options"])
                    choice = choice.replace("(", " (").replace(")", ") ")
                    choice = "Answer Choices:" + choice
                    q = json_res["question"].strip() + ' ' + choice
                    a = json_res["correct"]
                    id = 'temp_{}'.format(idx)
                elif args.dataset.lower() == 'commonsenseqa':
                    json_res = decoder.raw_decode(line)[0]
                    choice = "Answer Choices:"
                    for c in json_res["question"]["choices"]:
                        choice += " ("
                        choice += c["label"]
                        choice += ") "
                        choice += c["text"]
                    q = json_res["question"]["stem"].strip() + " " + choice
                    a = json_res["answerKey"]
                    id = 'temp_{}'.format(idx)
                else:
                    raise ValueError('not support dataset: {}'.format(args.dataset))
                questions.append(q)
                answers.append(a)
                ids.append(id)
    elif args.dataset.lower() in ['finqa', 'convfinqa']:
        with open(datapath) as f:
            json_data = json.load(f)
            for idx, line in enumerate(json_data):
                if args.dataset.lower() == 'convfinqa':
                    text = line['text'] + '\n'
                    table = line['table'].strip() + '\n'
                    q = 'Question: {}\n'.format(line['questions'])
                    a = line['answer']
                    id = 'temp_{}'.format(idx)
                elif args.dataset.lower() == 'finqa':
                    text = line['text'] + '\n'
                    table = line['table'].strip() + '\n'
                    q = 'Question: {}\n'.format(line['question'])
                    a = line['answer']
                    id = 'temp_{}'.format(idx)
                questions.append(text + table + q)
                answers.append(a)
                ids.append(id)
    else:
        raise ValueError('not support dataset: {}'.format(args.dataset))

    if args.test_end == 'full':
        return questions[int(args.test_start):], answers[int(args.test_start):], ids[int(args.test_start):], len(questions)
    else:
        return questions[int(args.test_start):int(args.test_end)], answers[
                                                                   int(args.test_start):int(args.test_end)], ids[
                                                                                                             int(args.test_start):int(
                                                                                                                 args.test_end)], int(args.test_end)
