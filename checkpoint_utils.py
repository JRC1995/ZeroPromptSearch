import logging
from os import fspath
from pathlib import Path
import pickle


def path_construct_fn(args, search_mode):
    dir = 'checkpoints/{}/'.format(args.dataset)
    Path(dir).mkdir(parents=True, exist_ok=True)
    if not search_mode:
        if args.SC:
            filename = dir + "{}_{}_SC.pkl".format(args.model, args.prompt_style)
        else:
            filename = dir + "{}_{}.pkl".format(args.model, args.prompt_style)
    else:
        filename = dir + "{}_{}_{}_{}.pkl".format(args.model,
                                                  args.prompt_style,
                                                  args.search_style,
                                                  args.reward_types)

    return filename


def save_checkpoint(args, state_dict):
    if args.search_style == "none":
        search_mode = False
    else:
        search_mode = True
    filename = path_construct_fn(args, search_mode)
    with open(filename, 'wb') as outfile:
        pickle.dump(state_dict, outfile)


def load_checkpoint(args):
    if args.search_style == "none":
        search_mode = False
    else:
        search_mode = True
    filename = path_construct_fn(args, search_mode)
    with open(filename, 'rb') as outfile:
        state_dict = pickle.load(outfile)
    return state_dict




