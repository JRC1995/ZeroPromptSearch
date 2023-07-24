import math
from node_transition import Node, node_transition
from model import generator as generator_class
from utils import argsort
import numpy as np
from typing import List
from node_transition import Node
from copy import deepcopy
import gc

def filter_too_bigs(nodes: List[Node], rewards: List[any], generator: generator_class):
    sentences = ["".join(node.chains) for node in nodes]
    batch_input_ids = generator.tokenizer(sentences)["input_ids"]
    candidates_ = []
    rewards_ = []
    for input_ids, node, reward in zip(batch_input_ids, nodes, rewards):
        if len(input_ids) <= 1500 or node.halt_status:
            candidates_.append(node)
            rewards_.append(reward)
    return candidates_, rewards_

def MultiGreedy(args: any, prompt: str, generator: generator_class) -> any:
    transition_engine = node_transition(generator=generator, args=args)
    root = Node()
    root.chains = [prompt]
    beams = [root]
    rewards = [0]
    completed = []
    max_depth = 18
    for d in range(max_depth):
        print("d: ", d)

        if d == 0:
            child_num = 20
        else:
            child_num = 5

        beam_children = transition_engine.batch_transition(nodes=beams,
                                                           child_num=child_num,
                                                           unique=False)
        if d == 0:
            beams = beam_children[0]
            rewards = [child.reward for child in beam_children[0]]
        else:
            beams_ = []
            rewards_ = []
            for i, children in enumerate(beam_children):
                child_rewards = [(rewards[i] + child.reward) for child in children]
                chosen_id = argsort(child_rewards)[-1]
                child = children[chosen_id]
                reward = child_rewards[chosen_id]
                beams_.append(child)
                rewards_.append(reward)
            beams = beams_
            rewards = rewards_
        print("beams_len: ", len(beams))

        beams, rewards = filter_too_bigs(nodes=beams,
                                         rewards=rewards,
                                         generator=generator)
        beams_ = []
        rewards_ = []
        for i, node in enumerate(beams):
            if node.halt_status:
                reward = math.exp(rewards[i] / node.root_distance)
                completed.append({"reward": reward, "prediction": "".join(node.chains[1:])})
                del node
            else:
                beams_.append(node)
                rewards_.append(rewards[i])

        beams = beams_
        rewards = rewards_

        print("completed_len: ", len(completed))
        print("beams_len: ", len(beams))

        if not beams:
            break

    if not completed:
        completed = [{"reward": 1, "prediction": ""}]

    print("candidates num: ", len(completed))

    return completed, prompt
