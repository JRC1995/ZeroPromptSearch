from model import generator as generator_class
from reward import batch_evaluate
import math


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
        prompt = prompt + "Step 1: "
        delimiter = "Step"
    elif args.prompt_style == "struct":
        prompt = prompt + "STEP 1 (Subproblem): "
        delimiter = "STEP"
    else:
        delimiter = "\n"

    Aid = select_opt_id(opt="(A)", tokenizer=generator.tokenizer)
    Bid = select_opt_id(opt="(B)", tokenizer=generator.tokenizer)

    batch_outputs = generator.generate(prompt=[prompt],
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
            reason_chains = (prompt + prediction).split(delimiter)
            meta_chains = []
            chains = []
            for i, reason_chain in enumerate(reason_chains):
                if i != 0:
                    chains = chains + [delimiter + reason_chain]
                else:
                    chains = [reason_chain]
                if reason_chain.strip() != "" and i != 0:
                    # print("chain: ", chain)
                    meta_chains.append(chains)

            # print("\n\nDONE\n\n")
            if args.prompt_style == "struct":
                meta_chains1 = []
                meta_chains2 = []
                for chains in meta_chains:
                    last_chain = chains[-1]
                    #print("chain: ", chain)
                    #print("last_chain: ", last_chain)
                    if "(Subproblem)" in last_chain:
                        meta_chains1.append(chains)
                    elif "(Solution)" in last_chain:
                        meta_chains2.append(chains)

                rewards1, num_rewards = batch_evaluate(generator=generator,
                                                       prompt_style=args.prompt_style,
                                                       batch_chains=meta_chains1,
                                                       positive_token_id=Aid,
                                                       negative_token_id=Bid,
                                                       substep_num=0,
                                                       reward_types=args.reward_types)

                rewards2, num_rewards = batch_evaluate(generator=generator,
                                                       prompt_style=args.prompt_style,
                                                       batch_chains=meta_chains2,
                                                       positive_token_id=Aid,
                                                       negative_token_id=Bid,
                                                       substep_num=1,
                                                       reward_types=args.reward_types)

                rewards = rewards1 + rewards2
            else:
                rewards, num_rewards = batch_evaluate(generator=generator,
                                                      prompt_style=args.prompt_style,
                                                      batch_chains=meta_chains,
                                                      positive_token_id=Aid,
                                                      negative_token_id=Bid,
                                                      substep_num=0,
                                                      reward_types=args.reward_types)
            rewards = [r / num_rewards for r in rewards]
            r = sum(rewards) / len(rewards)
            if "confidence" in args.reward_types:
                r = (confidence + r) / 2

        outputs.append({"reward": math.exp(r), "prediction": prediction})

    return outputs, prompt
