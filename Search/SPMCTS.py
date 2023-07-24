import math
from node_transition import Node, node_transition
import random
from model import generator as generator_class
from typing import Tuple, List
from utils import argmax


# main function for the Monte Carlo Tree Search
def SPMCTS(args: any, prompt: str, generator: generator_class, batch_size: int = 5, num_rounds: int = 4) -> any:
    transition_engine = node_transition(generator=generator, args=args, set_edge_links=True)
    root = Node()
    root.chains = [prompt]

    outputs = []
    for n in range(num_rounds):
        print("round: ", n)
        #print("root: ", root.chains)
        leaves = []
        depths = []
        for _ in range(batch_size):
            leaf, depth = traverse(root)
            leaves.append(leaf)
            depths.append(depth)

        final_states = batch_rollout(leaves, depths, transition_engine)

        for final_state in final_states:
            if final_state.parent:
                backpropagate(node=final_state.parent)
            accu_reward = calculate_path_reward(final_state)
            prediction = "".join(final_state.chains)
            outputs.append({"reward": accu_reward, "prediction": prediction})

    return outputs, prompt


# function for node traversal
def traverse(node) -> Tuple[Node, int]:

    def UCT(node):
        w = 1
        return node.Q + w * math.sqrt(math.log(node.parent.visited) / node.visited)

    depth = 0
    node.visited += 1
    while node.children:
        children = node.children
        depth += 1
        ucts = [UCT(child) for child in children]
        sum_uct = sum(ucts)
        ucts = [x / sum_uct for x in ucts]
        node = random.choices(population=children, weights=ucts, k=1)[0]
        node.visited += 1
    return node, depth

# function for the result of the simulation
def batch_rollout(nodes: List[Node],
                  depths: List[int],
                  transition_engine: generator_class) -> None:
    max_depth = 16
    completed = []
    while True:
        completed_, nodes, depths = filter_active(nodes, depths, max_depth, transition_engine.generator)
        completed = completed + completed_
        if nodes:
            nodes = rollout_policy(nodes, transition_engine)
            depths = [depth + 1 for depth in depths]
        else:
            break

    completed_ = []
    for node in completed:
        node.sum_reward = node.reward
        node.terminal_distance = 1
        node.Q = node.reward
        completed_.append(node)

    return completed_


def too_big(node: Node, generator: generator_class):
    sentence = "".join(node.chains)
    input_ids = generator.tokenizer.encode(sentence)
    # print("input_ids: ", input_ids)
    if len(input_ids) > 1500:
        return True
    return False


def filter_active(nodes: List[Node], depths: List[int], max_depth: int, generator: generator_class):
    completed = []
    active_nodes = []
    active_depths = []

    def terminal_check(node: Node, generator: generator_class):
        if node.halt_status or too_big(node, generator):
            return True
        else:
            return False

    for node, depth in zip(nodes, depths):
        if depth > max_depth:
            node.halt_status = True
        if terminal_check(node, generator):
            completed.append(node)
        else:
            active_nodes.append(node)
            active_depths.append(depth)

    return completed, active_nodes, active_depths


# function for randomly selecting a child node
def rollout_policy(nodes: List[Node], transition_engine: generator_class) -> Node:
    batch_children = transition_engine.batch_transition(nodes, child_num=5)
    selecteds = []
    for children in batch_children:
        rewards = [math.exp(child.reward) for child in children]
        rewards_sum = sum(rewards)
        reward_dist = [reward / rewards_sum for reward in rewards]
        selected = random.choices(population=children, weights=reward_dist, k=1)[0]
        selecteds.append(selected)
    return selecteds

# function for backpropagation
def backpropagate(node: Node) -> None:
    children = node.children
    sum_rewards = [child.sum_reward + node.reward for child in children]
    avg_rewards = [sum_reward / (child.terminal_distance + 1) for sum_reward, child in zip(sum_rewards, children)]
    id = argmax(avg_rewards)
    node.Q = math.exp(avg_rewards[id])
    node.terminal_distance = children[id].terminal_distance + 1
    node.sum_reward = sum_rewards[id]
    if node.parent:
        backpropagate(node=node.parent)


def calculate_path_reward(node: Node) -> float:
    accu_reward = node.reward
    depth = 1
    while node.parent:
        node = node.parent
        accu_reward += node.reward
        depth += 1
    accu_reward = math.exp(accu_reward / depth)
    return accu_reward
