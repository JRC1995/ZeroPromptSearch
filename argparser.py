import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', default='gsm8k',
        help="dataset",
        choices=["SVAMP", "gsm8k", "AQuA", "CommonsenseQA", "StrategyQA"]
    )
    parser.add_argument(
        "--prompt_style", default='ps', choices=["ps", "none", "cot", "cot_tab", "struct", "struct_min", "cot_step"]
    )
    parser.add_argument(
        "--search_style", default='BeamSearch', choices=["none",
                                                         "BeamSearch", "DivBeamSearch",
                                                         "MCTS", "SPMCTS",
                                                         "MultiSearch", "MultiGreedy"]
    )
    parser.add_argument(
        "--reward_types", default='confidence+correctness', choices=["both",
                                                                     "helpfulness",
                                                                     "correctness",
                                                                     "confidence",
                                                                     "confidence+both",
                                                                     "confidence+helpfulness",
                                                                     "confidence+correctness"]
    )
    parser.add_argument(
        "--model", default="LLAMA30_instruct", help="...",
        choices=["LLAMA30_instruct", "LLAMA60_instruct", "Redmond"]
    )
    parser.add_argument(
        "--test_start", default=0, type=int, help='number'
    )
    parser.add_argument(
        "--test_end", default="full", help='string, number'
    )
    parser.add_argument(
        "--gpu_ids", type=str, default="0", help='comma separated gpu ids that are to be kept visible. say: 0, 1'
    )
    parser.add_argument(
        "--SC", default=False, type=str2bool, help="self consistency"
    )
    parser.add_argument(
        "--checkpoint", default=False, type=str2bool, help="load checkpoint?"
    )
    parsed_args = parser.parse_args()

    if parsed_args.SC or parsed_args.search_style != "none":
        parsed_args.temperature = 0.8
    else:
        parsed_args.temperature = 0

    return parsed_args


args = parse_arguments()
