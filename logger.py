import logging
from os import fspath
from pathlib import Path

def get_logger(args):
    dir = 'logs/{}/'.format(args.dataset)
    Path(dir).mkdir(parents=True, exist_ok=True)

    if args.search_style == "none":
        if args.SC:
            filename = dir + "{}_{}_SC.log".format(args.model, args.prompt_style)
        else:
            filename = dir + "{}_{}.log".format(args.model, args.prompt_style)
    else:
        filename = dir + "{}_{}_{}_{}.log".format(args.model, args.prompt_style, args.search_style, args.reward_types)


    if args.checkpoint:
        logging.basicConfig(filename=filename,
                            format='%(asctime)s %(message)s',
                            filemode='a')
    else:
        logging.basicConfig(filename=filename,
                            format='%(asctime)s %(message)s',
                            filemode='w')

    # Creating an object
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    return logger
