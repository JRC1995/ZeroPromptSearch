import math
from model import generator as generator_class
from typing import List, Tuple


def format_prompt(chains: List[str], eval_prompt: str) -> str:
    return chains.strip() + eval_prompt

def batch_evaluate(generator: generator_class,
                   prompt_style: str,
                   batch_chains: List[List[str]],
                   positive_token_id: int, negative_token_id: int,
                   substep_num: int,
                   reward_types: str = "both") -> Tuple[List[float], int]:
    batch_prompt_state = ["".join(chains) for chains in batch_chains]
    b = len(batch_prompt_state)

    if prompt_style in ["struct", "struct_min"]:
        helpfulness = "\n\n# Let us pause for a moment and evaluate the last step before we proceed further.\n"\
                      "# Is solving the last subproblem helpful in making progress towards solving the main problem?\n" \
                      "# (A) Yes.\n# (B) No.\n# Answer: ("
        correctness = "\n\n# Let us pause for a moment and evaluate the last step before we proceed further.\n"\
                      "# Is the last calculation/reasoning step correct?\n" \
                      "# (A) Yes.\n# (B) No.\n# Answer: ("
        if substep_num == 0:
            eval_prompts = [helpfulness]
        else:
            eval_prompts = [correctness]
        batch_prompt_states = [batch_prompt_state]
    else:
        helpfulness = "\n\n# Let us pause for a moment and evaluate the last step before we proceed further.\n"\
                      "Question: Is the last step helpful?\n" \
                      "# (A) Yes.\n# (B) No.\n# Answer: ("
        correctness = "\n\n# Let us pause for a moment and evaluate the last step before we proceed further.\n"\
                      "# Question: Is the last calculation/reasoning step correct?\n" \
                      "# (A) Yes.\n# (B) No.\n# Answer: ("

        #batch_prompt_states = [batch_prompt_state]
        #eval_prompts = [correctness]
        if "both" in reward_types:
            eval_prompts = [helpfulness, correctness]
            batch_prompt_states = [batch_prompt_state, batch_prompt_state]
        elif "helpfulness" in reward_types:
            eval_prompts = [helpfulness]
            batch_prompt_states = [batch_prompt_state]
        elif "correctness" in reward_types:
            eval_prompts = [correctness]
            batch_prompt_states = [batch_prompt_state]
        else:
            raise ValueError("Invalid Reward Types")


    full_batch_prompt_states = []
    for eval_prompt, batch_prompt_state in zip(eval_prompts, batch_prompt_states):
        full_batch_prompt_states += [format_prompt(chains, eval_prompt) for chains in batch_prompt_state]

    #print("full_batch_prompt_states: ")

    batch_outputs = generator.generate(prompt=full_batch_prompt_states,
                                       max_length=1,
                                       num_samples=1,
                                       logprobs=1000,
                                       stop=[],
                                       temperature=0)

    prompt2pred = {}
    for output in batch_outputs:
        prompt2pred[output.prompt] = output.outputs

    rewards = []
    # print("eval_prompts: ", prompts)
    for id, prompt in enumerate(full_batch_prompt_states):

        sample = prompt2pred[prompt][0]
        logprobs = sample.logprobs[0]
        pos_prob = 0
        neg_prob = 0
        if negative_token_id in logprobs:
            neg_prob = math.exp(logprobs[negative_token_id])
        if positive_token_id in logprobs:
            pos_prob = math.exp(logprobs[positive_token_id])

        pos_prob = 0 if pos_prob == 0 else pos_prob / (pos_prob + neg_prob)

        if pos_prob == 0:
            reward = -99999
        else:
            reward = math.log(pos_prob)

        #print("Reward Prompt: ", full_batch_prompt_states[id])
        #print("Gen: ", sample.text)
        #print("Pos prob: ", pos_prob)

        rewards.append(reward)

    sum_rewards = []
    num_eval_prompts = len(eval_prompts)
    for i in range(b):
        sum_reward = sum([rewards[i + (j * b)] for j in range(len(eval_prompts))])
        sum_rewards.append(sum_reward)

    assert len(sum_rewards) == b

    return sum_rewards, num_eval_prompts
