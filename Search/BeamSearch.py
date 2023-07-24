import math
from node_transition import Node, node_transition
from model import generator as generator_class
from utils import argsort
import numpy as np
from typing import List
from node_transition import Node
from copy import deepcopy
import gc


def terminal(nodes: List[Node], beam_size: int) -> bool:
    halt_count = 0
    for node in nodes:
        if node.halt_status:
            halt_count += 1
    if halt_count < beam_size:
        return False
    else:
        return True


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


# main function for the Monte Carlo Tree Search
def BeamSearch(args: any, prompt: str, generator: generator_class, beam_size: int = 5) -> any:
    transition_engine = node_transition(generator=generator, args=args)
    root = Node()
    root.chains = [prompt]
    beams = [root]
    rewards = [0]
    completed = []
    max_depth = 18
    temperature = 0.5
    alpha = 0.5
    for d in range(max_depth):
        print("d: ", d)
        assert len(beams) <= beam_size

        child_num = 16
        beam_children = transition_engine.batch_transition(nodes=beams, child_num=child_num)
        all_rewards = []
        all_children = []
        for i, children in enumerate(beam_children):
            all_rewards = all_rewards + [rewards[i] + deepcopy(child.reward) for child in children]
            all_children = all_children + children
        # print("all_children_len: ", len(all_children))
        # print("all_rewards: ", all_rewards)

        if d == (max_depth - 1):
            terminal_flag = True
        else:
            all_children, all_rewards = filter_too_bigs(nodes=all_children,
                                                        rewards=all_rewards,
                                                        generator=generator)
            terminal_flag = terminal(nodes=all_children,
                                     beam_size=beam_size)

        if len(all_children) > beam_size and (not terminal_flag):
            """
            reward_probs = [math.exp(r / (n.root_distance * temperature)) for r, n in
                            zip(all_rewards, all_children)]
            """
            reward_probs = [math.exp(r / (temperature)) for r, n in
                            zip(all_rewards, all_children)]

            denom = sum(reward_probs)
            if denom == 0:
                reward_dist = np.asarray([1 / len(all_rewards)] * len(reward_probs))
            else:
                reward_dist = np.asarray(reward_probs) / denom
            idx = [i for i in range(len(all_rewards))]
            chosen_idx = np.random.choice(a=np.asarray(idx),
                                          p=reward_dist,
                                          size=beam_size,
                                          replace=False).tolist()

            assert len(chosen_idx) == beam_size
            beams = [all_children[id] for id in chosen_idx]
            rewards = [all_rewards[id] for id in chosen_idx]

            chosen_idx = {i: 1 for i in chosen_idx}
            for i, node in enumerate(all_children):
                if i not in chosen_idx:
                    del node
        else:
            beams = all_children
            rewards = all_rewards

        if terminal_flag:
            completed = []
            for i, node in enumerate(beams):
                if node.halt_status:
                    # reward = math.exp(rewards[i] / node.root_distance)
                    reward = math.exp(rewards[i])
                    completed.append({"reward": reward, "prediction": "".join(node.chains[1:])})
                    del node
            break

        temperature = temperature * alpha

    if not completed:
        completed = [{"reward": 1, "prediction": ""}]

    print("candidates num: ", len(completed))

    return completed, prompt
