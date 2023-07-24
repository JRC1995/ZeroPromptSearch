from model import generator as generator_class

def none(args: any, prompt: str, generator: generator_class) -> any:
    if args.prompt_style == "cot_step":
        prompt = prompt + "Step 1: "
    elif args.prompt_style == "struct":
        prompt = prompt + "STEP 1 (Subproblem): "

    if args.SC:
        n = 20
    else:
        n = 1

    batch_outputs = generator.generate(prompt=[prompt],
                                       max_length=1000,
                                       temperature=args.temperature,
                                       num_samples=n)
    outputs = [{"prediction": output.text, "reward": 1} for output in batch_outputs[0].outputs]
    return outputs, prompt
