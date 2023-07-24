from model import generator as generator_class
from reward import batch_evaluate
import math
import copy


def get_confidence(sample):
    logprobs = sample.logprobs
    confidence = 0
    for logprob in logprobs:
        for k in logprob:
            confidence += logprob[k]
            break
    return confidence / len(logprobs)


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


# main function for the Monte Carlo Tree Search
def MultiSearch(args: any, prompt: str, generator: generator_class) -> any:
    if args.prompt_style == "cot_step":
        ext = "Step 1: "
        ext_prompt = prompt + ext
        delimiter = "Step"
    elif args.prompt_style == "struct":
        ext = "STEP 1 (Subproblem): "
        ext_prompt = prompt + ext
        delimiter = "STEP"
    else:
        ext = ""
        ext_prompt = prompt
        delimiter = "\n"

    Aid = select_opt_id(opt="(A)", tokenizer=generator.tokenizer)
    Bid = select_opt_id(opt="(B)", tokenizer=generator.tokenizer)

    batch_outputs = generator.generate(prompt=[ext_prompt],
                                       max_length=1000,
                                       temperature=args.temperature,
                                       num_samples=20,
                                       logprobs=1)
    predictions = [output.text for output in batch_outputs[0].outputs]
    confidences = [get_confidence(output) for output in batch_outputs[0].outputs]

    outputs = []
    for prediction, confidence in zip(predictions, confidences):
        if args.reward_types == "confidence":
            r = confidence
        else:
            reason_chains = [prompt] + (ext + prediction).split(delimiter)
            #print("reason_chains: ", reason_chains)
            meta_chains = []
            chains = []
            chain = ""
            steps = []
            step = 1
            chain_len = len(reason_chains)
            for i, reason_chain in enumerate(reason_chains):
                if i == 0:
                    #print("chain: ", reason_chain)
                    chains = [reason_chain]
                else:
                    if args.prompt_style not in ["struct", "struct_min", "cot_step"]:
                        if reason_chain.strip() == "":
                            chain += reason_chain + delimiter
                        else:
                            if i == (chain_len-1):
                                chain = chain + reason_chain
                            else:
                                chain = chain + reason_chain + delimiter
                            #print("chain: ", chain)
                            chains.append(chain)
                            meta_chains.append(copy.deepcopy(chains))
                            steps.append(step)
                            step += 1
                            chain = ""
                    else:
                        if reason_chain.strip() != "":
                            #print("chain: ", delimiter + reason_chain)
                            chains.append(delimiter + reason_chain)
                            #print("chains: ", chains)
                            meta_chains.append(copy.deepcopy(chains))
                            steps.append(step)
                            step += 1


            #print("\n\nDONE\n\n")
            if args.prompt_style in ["struct", "struct_min"]:
                meta_chains1 = []
                meta_chains2 = []
                steps1 = []
                steps2 = []
                step1 = 1
                step2 = 1
                for chains in meta_chains:
                    last_chain = chains[-1]
                    #print("chains: ", chains)
                    #print("last_chain: ", last_chain)
                    if "(Subproblem)" in last_chain:
                        meta_chains1.append(chains)
                        steps1.append(step1)
                        step1 += 1
                    elif "(Solution)" in last_chain:
                        meta_chains2.append(chains)
                        steps2.append(step2)
                        step2 += 1

                rewards1, num_rewards = batch_evaluate(generator=generator,
                                                       prompt_style=args.prompt_style,
                                                       batch_chains=meta_chains1,
                                                       positive_token_id=Aid,
                                                       negative_token_id=Bid,
                                                       substep_num=0,
                                                       step_nums=steps1,
                                                       reward_types=args.reward_types)

                rewards2, num_rewards = batch_evaluate(generator=generator,
                                                       prompt_style=args.prompt_style,
                                                       batch_chains=meta_chains2,
                                                       positive_token_id=Aid,
                                                       negative_token_id=Bid,
                                                       substep_num=1,
                                                       step_nums=steps2,
                                                       reward_types=args.reward_types)

                rewards = rewards1 + rewards2
            else:
                rewards, num_rewards = batch_evaluate(generator=generator,
                                                      prompt_style=args.prompt_style,
                                                      batch_chains=meta_chains,
                                                      positive_token_id=Aid,
                                                      negative_token_id=Bid,
                                                      substep_num=0,
                                                      step_nums=steps,
                                                      reward_types=args.reward_types)
            rewards = [r / num_rewards for r in rewards]
            r = sum(rewards) / len(rewards)
            if "confidence" in args.reward_types:
                r = (confidence + r) / 2

        outputs.append({"reward": math.exp(r), "prediction": prediction})

    return outputs, ext_prompt
