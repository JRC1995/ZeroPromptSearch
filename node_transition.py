from transformers import AutoTokenizer, LlamaTokenizer
from typing import List, Optional, Union
from copy import deepcopy
from reward import batch_evaluate
from model import generator as generator_class
import math

"""
Class for Nodes
"""


class Node:
    def __init__(self):
        self.chains = []
        self.parent = None
        self.children = []
        self.substep_num = -1
        self.reward = 0
        self.Q = 0
        self.sum_reward = 0
        self.step_num = 0
        self.halt_status = False
        self.visited = 1
        self.root_distance = 0
        self.terminal_distance = 0


def select_opt_id(opt: str, tokenizer: any) -> int:
    idx = tokenizer.encode(opt)
    flag = False
    for id in idx:
        if flag:
            return id
        else:
            token = tokenizer.decode(id)
            if token == "(":
                flag = True
    return idx[0]


class node_transition:
    def __init__(self, generator: generator_class,
                 args: any,
                 max_length: int = 512,
                 set_edge_links: bool = False):
        self.generator = generator
        self.max_length = max_length
        self.temperature = args.temperature
        self.Aid = select_opt_id(opt="(A)", tokenizer=generator.tokenizer)
        self.Bid = select_opt_id(opt="(B)", tokenizer=generator.tokenizer)
        self.reward = 0
        self.prompt_style = args.prompt_style
        self.set_edge_links = set_edge_links
        self.reward_types = args.reward_types

    def prompt_format(self, node):
        prompt_state = "".join(node.chains)  # combine reasoning steps (includes initial question prompt)

        if self.prompt_style in ["struct", "struct_min"]:
            # experimental prompt instructed to decompose into a subproblem, then list facts, then solution
            # each step have 3 substeps - 0) Subproblem creation 1) list relevant facts 2) Solution
            # we treat the last two substeps as a single substep for reward -- thus, substep number varies in a modulo 2 fashion
            # struct_min removes the "list relevant facts" substep.
            substep_num = int((node.substep_num + 1) % 2)
            if substep_num == 0:
                step = node.step_num + 1
            else:
                step = node.step_num  # if substep_num non-zero it means we are in the same step (so we keep the same step_num)
            if substep_num == 0:
                step_prompt = "STEP {} (Subproblem): ".format(step)
            else:
                if self.prompt_style == "struct":
                    step_prompt = "STEP {} (Facts): ".format(step)
                else:
                    # if struct_mini
                    step_prompt = "STEP {} (Solution): ".format(step)

            return {"prompt_state": prompt_state + step_prompt,
                    "step_prompt": step_prompt,
                    "substep_num": substep_num,
                    "step": step}

        elif self.prompt_style == "cot_step":
            step = node.step_num + 1
            step_prompt = "Step {}: ".format(step)
            return {"prompt_state": prompt_state + step_prompt,
                    "step_prompt": step_prompt,
                    "substep_num": None,
                    "step": step}
        else:
            step = node.step_num + 1
            return {"prompt_state": prompt_state,
                    "step_prompt": "",
                    "substep_num": None,
                    "step": step}

    def get_confidence(self, sample):
        logprobs = sample.logprobs
        confidence = 0
        for logprob in logprobs:
            for k in logprob:
                confidence += logprob[k]
                break
        return confidence / len(logprobs)

    def child_format(self, parent_node: Node, outputs: any,
                     step_prompt: str, step: int, substep_num: int, unique: bool) -> List[Node]:
        if parent_node.halt_status:
            parent_node.reward = 0
            return [parent_node]
        else:
            new_reason_chains = [output.text for output in outputs]
            # print("new_reason_chains: ", new_reason_chains)
            confidences = [self.get_confidence(output) for output in outputs]
            children = []
            dict_chains = {}
            for confidence, reason_chain in zip(confidences, new_reason_chains):
                if reason_chain not in dict_chains:
                    if unique:
                        dict_chains[reason_chain] = 1
                    new_node = Node()
                    new_node.parent = parent_node
                    if self.prompt_style in ["struct", "struct-mini"]:
                        if substep_num == 1:
                            flag_str = "STEP {} ".format(step + 1)
                            if reason_chain[-len(flag_str):] == flag_str:
                                reason_chain = reason_chain[0:-len(flag_str)]
                            else:
                                new_node.halt_status = True
                            reason_chain = reason_chain.strip() + "\n\n"
                        else:
                            if reason_chain[-2:] == "ST":
                                reason_chain = reason_chain[0:-2]
                            reason_chain = reason_chain.rstrip() + "\n\n"
                    elif self.prompt_style == "cot_step":
                        flag_str = "St"
                        if reason_chain[-len(flag_str):] == flag_str:
                            reason_chain = reason_chain[0:-len(flag_str)]
                        else:
                            new_node.halt_status = True
                        reason_chain = reason_chain.rstrip() + "\n\n"

                    reason_chain = step_prompt + reason_chain
                    new_node.chains = parent_node.chains + [reason_chain]
                    new_node.step_num = step
                    new_node.substep_num = substep_num
                    if self.set_edge_links:
                        new_node.parent = parent_node
                    new_node.reward = confidence
                    children.append(new_node)
            # print("\n\n\n")
            # parent_node.children = children_

            if not (self.prompt_style in ["struct", "struct_min", "cot_step"]):
                batch_chains = ["".join(child.chains) for child in children]
                batch_outputs = self.generator.generate(prompt=batch_chains,
                                                        max_length=10,
                                                        num_samples=1,
                                                        stop=[],
                                                        temperature=0)
                chain2pred = {}
                for output in batch_outputs:
                    chain2pred[output.prompt] = output.outputs[0].text
                batch_outputs = [chain2pred[chain] for chain in batch_chains]

                children_ = []
                for node, text in zip(children, batch_outputs):
                    if text.strip() == "":
                        node.halt_status = True
                        node.chains = node.chains[0:-1] + [node.chains[-1] + "\n\n"]
                    else:
                        last_chain = node.chains[-1]
                        for c in text:
                            if c == "\n":
                                last_chain = last_chain + "\n"
                            else:
                                break
                        node.chains = node.chains[0:-1] + [last_chain]
                    children_.append(node)
                children = children_

            batch_chains = [child.chains for child in children]

            if self.reward_types == "confidence":
                rewards = [0] * len(children)
                num_rewards = 0
            else:
                rewards, num_rewards = batch_evaluate(generator=self.generator,
                                                      prompt_style=self.prompt_style,
                                                      batch_chains=batch_chains,
                                                      positive_token_id=self.Aid,
                                                      negative_token_id=self.Bid,
                                                      step_nums=[step] * len(batch_chains),
                                                      substep_num=substep_num,
                                                      reward_types=self.reward_types)

            children_ = []
            for child, reward in zip(children, rewards):
                if "confidence" in self.reward_types:
                    child.reward = (child.reward + reward) / (1 + num_rewards)
                else:
                    child.reward = (reward) / (num_rewards)
                child.sum_reward = child.reward
                child.root_distance = parent_node.root_distance + 1
                children_.append(child)

        if self.set_edge_links:
            child_dict = {}
            if parent_node.children:
                for child in parent_node.children:
                    chain = "".join(child.chains)
                    child_dict[chain] = child
            for child in children_:
                chain = "".join(child.chains)
                child_dict[chain] = child
            parent_node.children = [v for k, v in child_dict.items()]

        return children_

    def batch_transition(self, nodes: List[Node], child_num: int, unique: bool = True) -> List[List[Node]]:
        prompt_state_dicts = [self.prompt_format(node) for node in nodes]
        prompt_states = [x["prompt_state"] for x in prompt_state_dicts]

        if (self.prompt_style == "struct") or (self.prompt_style == "struct_min"):
            substep_num = prompt_state_dicts[0]["substep_num"]
            if substep_num == 0:
                batch_outputs = self.generator.generate(prompt=prompt_states,
                                                        max_length=self.max_length,
                                                        num_samples=child_num,
                                                        stop=["EP", "###"],
                                                        logprobs=1,
                                                        temperature=self.temperature)
            else:
                batch_outputs = self.generator.generate(prompt=prompt_states,
                                                        max_length=self.max_length,
                                                        num_samples=child_num,
                                                        stop=["(Subproblem):", "###"],
                                                        logprobs=1,
                                                        temperature=self.temperature)
        elif self.prompt_style == "cot_step":
            step_num = prompt_state_dicts[0]["step"]
            batch_outputs = self.generator.generate(prompt=prompt_states,
                                                    max_length=self.max_length,
                                                    num_samples=child_num,
                                                    stop=["ep {}:".format(step_num + 1), "###"],
                                                    logprobs=1,
                                                    temperature=self.temperature)
        else:
            batch_outputs = self.generator.generate(prompt=prompt_states,
                                                    max_length=self.max_length,
                                                    num_samples=child_num,
                                                    stop=["\n", "###"],
                                                    logprobs=1,
                                                    temperature=self.temperature)

        prompt2pred = {}
        for output in batch_outputs:
            prompt2pred[output.prompt] = output.outputs

        batch_children = []
        for prompt, node, prompt_state_dict in zip(prompt_states, nodes, prompt_state_dicts):
            # print("parent prompt state: ", prompt_state_dict["prompt_state"])
            # print("\n\n")
            children = self.child_format(parent_node=node,
                                         outputs=prompt2pred[prompt],
                                         unique=unique,
                                         step_prompt=prompt_state_dict["step_prompt"],
                                         step=prompt_state_dict["step"],
                                         substep_num=prompt_state_dict["substep_num"])
            batch_children.append(children)

        return batch_children
