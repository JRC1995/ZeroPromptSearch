import math
from node_transition import Node, node_transition
from model import generator as generator_class
from utils import argsort
import numpy as np
from typing import List, Tuple
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


def filter_too_bigs(nodes: List[Node],
                    rewards: List[float],
                    generator: generator_class) -> Tuple[List[Node], List[float]]:
    sentences = ["".join(node.chains) for node in nodes]
    batch_input_ids = generator.tokenizer(sentences)["input_ids"]
    candidates_ = []
    rewards_ = []
    for input_ids, node, reward in zip(batch_input_ids, nodes, rewards):
        if len(input_ids) <= 1500 or node.halt_status:
            candidates_.append(node)
            rewards_.append(reward)
    return candidates_, rewards_


def sample(reward_probs: List[float], sample_num: int) -> List[int]:
    denom = sum(reward_probs)
    if denom == 0:
        reward_dist = np.asarray([1 / len(reward_probs)] * len(reward_probs))
    else:
        reward_dist = np.asarray(reward_probs) / denom
    idx = [i for i in range(len(reward_probs))]
    chosen_idx = np.random.choice(a=np.asarray(idx),
                                  p=reward_dist,
                                  size=sample_num,
                                  replace=False).tolist()
    return chosen_idx


# main function for the Monte Carlo Tree Search
def DivBeamSearch(args: any, prompt: str, generator: generator_class, beam_size: int = 5) -> any:
    transition_engine = node_transition(generator=generator, args=args, set_edge_links=True)
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
            reward_probs = [math.exp(r / (temperature)) for r, n in
                            zip(all_rewards, all_children)]

            priority_idx = argsort(reward_probs)
            priority_idx.reverse()
            priority_children = []
            priority_parent_dict = {}
            non_priority_children = []

            for i in priority_idx:
                child = all_children[i]
                parent_chain = "".join(child.parent.chains)
                priority_parent_dict[parent_chain] = priority_parent_dict.get(parent_chain, 0) + 1
                if priority_parent_dict[parent_chain] <= 2:
                    priority_children.append(i)
                else:
                    non_priority_children.append(i)

            priority_probs = [reward_probs[i] for i in priority_children]
            non_priority_probs = [reward_probs[i] for i in non_priority_children]

            priority_chosen_idx = sample(priority_probs, min(beam_size, len(priority_probs)))
            if len(priority_chosen_idx) >= beam_size:
                chosen_idx = [priority_children[i] for i in priority_chosen_idx]
            else:
                left_size = (beam_size - len(priority_chosen_idx))
                non_priority_chosen_idx = sample(non_priority_probs, left_size)
                chosen_idx = [priority_children[i] for i in priority_chosen_idx] \
                             + [non_priority_children[i] for i in non_priority_chosen_idx]

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
