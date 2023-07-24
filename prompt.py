def set_prompt(args, question):
    ps_prompt = "Let's first understand the problem, extract relevant variables and their corresponding numerals, " \
                "and devise a plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to " \
                "correct numeral calculation and commonsense), solve the problem step by step, and show the answer. "

    cot_prompt = "Let's think step by step. "

    none_prompt = ""

    cot_step_prompt = "Let's think step by step.\n\n"

    cot_tab_prompt = "\n|step|subquestion|process|result|\n"

    struct_min_prompt = "Let us solve the problem step by step. In each step n, I will take the following 2 substeps:\n" \
                        "  1. Following 'STEP n (Subproblem): ', I will identify an elementary subproblem that is needed to be solved for the main problem.\n" \
                        "  2. Following `STEP n (Solution): ', I will devise and follow a step-by-step procedure to solve the subproblem.\n" \
                        "If there are more subproblems to solve, I will start STEP n+1 with another subproblem. " \
                        "If there are no more subproblems to solve, I will synthesize the solution to the main problem given the solutions to the subproblems.\n\n"

    struct_prompt = "Let us solve the problem step by step. In each step n, I will take the following 3 substeps:\n" \
                    "  1. Following 'STEP n (Subproblem): ', I will identify an elementary subproblem that is needed to be solved for the main problem.\n" \
                    "  2. Following 'STEP n (Facts): ', I will list the known facts and variables needed to solve the subproblem.\n" \
                    "  3. Following `STEP n (Solution): ', I will devise and follow a step-by-step procedure to solve the subproblem.\n" \
                    "If there are more subproblems to solve, I will start STEP n+1 with another subproblem. " \
                    "If there are no more subproblems to solve, I will synthesize the solution to the main problem given the solutions to the subproblems.\n\n"

    instruction = eval(args.prompt_style + "_prompt")

    if args.model in ["LLAMA30_instruct", "LLAMA60_instruct"]:
        system = "You are an AI assistant that helps people find information. " \
                 "User will you give you a question. Your task is to answer as faithfully as you can. " \
                 "While answering think step-by-step and justify your answer."

        prompt = f"### System: {system}\n\n### User: {question}\n\n### Assistant: {instruction}"
    elif args.model in ["Redmond"]:
        system = "### human: Interact in conversation to the best of your ability, please be logical, intelligent and coherent.\n\n"\
        "### response: Sure! sounds good.\n\n"

        prompt = f"{system}\n\n### human: {question}\n\n### response: {instruction}"
    else:
        prompt = f"Q: {question}\n\nA: {instruction}"
    return prompt
