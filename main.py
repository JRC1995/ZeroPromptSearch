from argparser import args
import os

os.environ["TRANSFORMERS_CACHE"] = "cache/"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
from tqdm import tqdm
from answer_extractor import extract_answer
from utils import load_data, argsort
from model import generator
from Search import *
from prompt import set_prompt
from evaluator import evaluate
from logger import get_logger
from checkpoint_utils import *
import time

if __name__ == '__main__':

    tensor_parallel_size = len(args.gpu_ids.split(","))
    logger = get_logger(args)

    generator = generator(model_name=args.model, tensor_parallel_size=tensor_parallel_size)  # load generator
    search = eval(
        args.search_style)  # load search wrapper function (search function can be MCTS or beam search or whatever)


    def print_and_log(msg, logger):
        logger.info(msg + "\n")
        print(msg)


    end_flag = False
    if args.checkpoint:
        state_dict = load_checkpoint(args)
        voted_correct = state_dict["voted_correct"]
        reward_voted_correct = state_dict["reward_voted_correct"]
        max_reward_correct = state_dict["max_reward_correct"]
        topk_reward_voted_correct = state_dict["topk_reward_voted_correct"]
        total = state_dict["total"]
        total_time = state_dict["total_time"]
        start_id = state_dict["start_id"]
        if start_id is None:
            # we set start_id to none when the start_id exceeds datasize length. This indicates we are at the end.
            end_flag = True
            # end_flag == True implies that dataset has been already fully traversed during checkpoint
        else:
            args.test_start = state_dict["start_id"]
    else:
        voted_correct = 0
        reward_voted_correct = 0
        max_reward_correct = 0
        topk_reward_voted_correct = 0
        total = 0
        total_time = 0
        start_id = int(args.test_start)

    if not end_flag:
        questions, answers, ids, dataset_size = load_data(args)  # load dataset
        max_reward_generation = ""
        interval = 5  # interval for printing moving averages of metrics
        i = 0
        with tqdm(total=len(questions), desc="generating", position=0) as pbar:
            for id, question in enumerate(questions):
                answer = answers[id]
                init_time = time.perf_counter()
                prompt = set_prompt(args, question)

                """
                Main Search Procedure
                """
                outputs, prompt = search(generator=generator, prompt=prompt, args=args)
                predictions = [x["prediction"] for x in outputs]
                rewards = [x["reward"] for x in outputs]

                """
                Extract Answers
                """
                pred_answers, full_predictions = extract_answer(args, prompt, predictions, generator)

                """
                Answering Voting in Different Manners
                """
                voted_answer = ""  # variable for majority voted answer
                reward_voted_answer = ""  # variable for reward-weighted majority voted answer
                topk_reward_voted_answer = ""
                max_reward_answer = ""  # variable for maximum reward answer
                max_votes = -1  # to track maximum votes received by a single answer
                max_reward = -1  # to track maximum reward
                max_topk_reward_votes = -1
                max_reward_votes = -1  # to track maximum reward weighted votes received by a single answer
                answer2votes = {}
                answer2reward_votes = {}
                answer2topk_reward_votes = {}

                topk_idx = argsort(rewards)[-5:]
                topk_rewards = [rewards[x] for x in topk_idx]
                topk_pred_answers = [pred_answers[x] for x in topk_idx]

                pid = 0
                for pred_answer, reward, generation in zip(pred_answers, rewards, full_predictions):
                    if pred_answer is not None:

                        answer2votes[pred_answer] = answer2votes.get(pred_answer, 0) + 1
                        answer2reward_votes[pred_answer] = answer2reward_votes.get(pred_answer, 0) + reward
                        if pid in topk_idx:
                            answer2topk_reward_votes[pred_answer] \
                                = answer2topk_reward_votes.get(pred_answer, 0) + reward

                        votes = answer2votes[pred_answer]
                        reward_votes = answer2reward_votes[pred_answer]
                        if pid in topk_idx:
                            topk_reward_votes = answer2topk_reward_votes[pred_answer]
                            if topk_reward_votes > max_topk_reward_votes:
                                max_topk_reward_votes = topk_reward_votes
                                topk_reward_voted_answer = pred_answer

                        if votes > max_votes:
                            max_votes = votes
                            voted_answer = pred_answer
                        if reward_votes > max_reward_votes:
                            max_reward_votes = reward_votes
                            reward_voted_answer = pred_answer
                        if reward > max_reward:
                            max_reward = reward
                            max_reward_answer = pred_answer
                            max_reward_generation = generation

                    pid += 1

                if voted_answer == "":
                    voted_answer = None
                    reward_voted_answer = None
                    topk_reward_voted_answer = None
                    max_reward_answer = None

                """
                Evaluate Answers
                """
                voted_value = evaluate(args, voted_answer, answer)
                reward_voted_value = evaluate(args, reward_voted_answer, answer)
                max_reward_value = evaluate(args, max_reward_answer, answer)
                topk_reward_voted_value = evaluate(args, topk_reward_voted_answer, answer)

                """
                Print, Log, and Updates
                """
                final_time = time.perf_counter()
                iter_time = final_time - init_time
                total_time += iter_time

                print_and_log("Dataset: {}; Search style: {}; model: {}; reward_types: {}" \
                              .format(args.dataset, args.search_style, args.model, args.reward_types), logger)
                print_and_log("QID: {}".format(start_id + id), logger)
                print_and_log("Prompt (style {}): {}".format(args.prompt_style, prompt), logger)
                # print("\nGeneration: ", full_predictions)
                print_and_log("\nMax Reward Generation: {}".format(max_reward_generation), logger)
                print_and_log("\nPred Answers: {}".format(pred_answers), logger)
                print_and_log("\nRewards: {}".format(rewards), logger)
                print_and_log("\nTop K Pred Answers: {}".format(topk_pred_answers), logger)
                print_and_log("\nTop K Rewards: {}".format(topk_rewards), logger)
                print_and_log("\nVoted Answer: {}".format(voted_answer), logger)
                print_and_log("^Correctness: {}".format(voted_value), logger)
                print_and_log("\nReward Voted Answer: {}".format(reward_voted_answer), logger)
                print_and_log("^Correctness: {}".format(reward_voted_value), logger)
                print_and_log("\nTop K Reward Voted Answer: {}".format(topk_reward_voted_answer), logger)
                print_and_log("^Correctness: {}".format(topk_reward_voted_value), logger)
                print_and_log("\nMax Reward Answer: {}".format(max_reward_answer), logger)
                print_and_log("^Correctness: {}".format(max_reward_value), logger)
                print_and_log("\nTrue Answer: {}".format(answer), logger)
                print_and_log("\nTime Taken: {}".format(iter_time), logger)
                print_and_log("Total Time Taken: {}".format(total_time), logger)

                voted_correct += voted_value
                reward_voted_correct += reward_voted_value
                max_reward_correct += max_reward_value
                topk_reward_voted_correct += topk_reward_voted_value
                total += 1

                if total % interval == 0:
                    print_and_log("\n\n--------------\nMoving Majority Voted Accuracy: {}%" \
                                  .format((voted_correct / total) * 100), logger)
                    print_and_log("Moving Reward Voted Accuracy: {}%" \
                                  .format((reward_voted_correct / total) * 100), logger)
                    print_and_log("Moving Top K Reward Voted Accuracy: {}%" \
                                  .format((topk_reward_voted_correct / total) * 100), logger)
                    print_and_log("Moving Max Reward Accuracy: {}%\n--------------\n\n" \
                                  .format((max_reward_correct / total) * 100), logger)

                if start_id + id + 1 > dataset_size - 1:
                    # If we are here it means that we are at the last dataset
                    start_id_ = None
                    # end_flat will be enabled as True if checkpoint at this point is loaded
                else:
                    start_id_ = start_id + id + 1

                state_dict = {"voted_correct": voted_correct,
                              "reward_voted_correct": reward_voted_correct,
                              "topk_reward_voted_correct": topk_reward_voted_correct,
                              "max_reward_correct": max_reward_correct,
                              "total": total,
                              "total_time": total_time,
                              "start_id": start_id_}

                save_checkpoint(args, state_dict=state_dict)
                print("\nCheckpoint Saved\n")
                pbar.update(1)

    print_and_log("\n\nFinal Majority Voted Accuracy: {}%" \
                  .format((voted_correct / total) * 100), logger)
    print_and_log("Final Reward Voted Accuracy: {}%" \
                  .format((reward_voted_correct / total) * 100), logger)
    print_and_log("Final Top K Reward Voted Accuracy: {}%" \
                  .format((topk_reward_voted_correct / total) * 100), logger)
    print_and_log("Final Max Reward Accuracy: {}%" \
                  .format((max_reward_correct / total) * 100), logger)
