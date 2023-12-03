## Zero-Shot Prompts for Step Decomposition and Search

This is more of an experimental/research project. It implements a prompting pipeline combined with a wrapper for auto-decomposing steps and searches through the "step space" (eg. by beam search, MCTS, etc.) guided by self-evaluation.

### Credits:

Some of the data-extraction/answer-extraction codes (```utils.py``` and ```answer_extraction.py```) are adapted from: https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting

Dataset References:
* GSM8K:
```
  @misc{cobbe2021training,
      title={Training Verifiers to Solve Math Word Problems}, 
      author={Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Mark Chen and Heewoo Jun and Lukasz Kaiser and Matthias Plappert and Jerry Tworek and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and John Schulman},
      year={2021},
      eprint={2110.14168},
      archivePrefix={arXiv},
      primaryClass={cs.LG}}
```
* SVAMP
```
@inproceedings{patel-etal-2021-nlp,
    title = "Are {NLP} Models really able to Solve Simple Math Word Problems?",
    author = "Patel, Arkil  and
      Bhattamishra, Satwik  and
      Goyal, Navin",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.168",
    doi = "10.18653/v1/2021.naacl-main.168",
    pages = "2080--2094",
    abstract = "The problem of designing NLP solvers for math word problems (MWP) has seen sustained research activity and steady gains in the test accuracy. Since existing solvers achieve high performance on the benchmark datasets for elementary level MWPs containing one-unknown arithmetic word problems, such problems are often considered {``}solved{''} with the bulk of research attention moving to more complex MWPs. In this paper, we restrict our attention to English MWPs taught in grades four and lower. We provide strong evidence that the existing MWP solvers rely on shallow heuristics to achieve high performance on the benchmark datasets. To this end, we show that MWP solvers that do not have access to the question asked in the MWP can still solve a large fraction of MWPs. Similarly, models that treat MWPs as bag-of-words can also achieve surprisingly high accuracy. Further, we introduce a challenge dataset, SVAMP, created by applying carefully chosen variations over examples sampled from existing datasets. The best accuracy achieved by state-of-the-art models is substantially lower on SVAMP, thus showing that much remains to be done even for the simplest of the MWPs.",
}
```
* AQuA:
```
@inproceedings{ling-etal-2017-program,
    title = "Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems",
    author = "Ling, Wang  and
      Yogatama, Dani  and
      Dyer, Chris  and
      Blunsom, Phil",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P17-1015",
    doi = "10.18653/v1/P17-1015",
    pages = "158--167",
    abstract = "Solving algebraic word problems requires executing a series of arithmetic operations{---}a program{---}to obtain a final answer. However, since programs can be arbitrarily complicated, inducing them directly from question-answer pairs is a formidable challenge. To make this task more feasible, we solve these problems by generating answer rationales, sequences of natural language and human-readable mathematical expressions that derive the final answer through a series of small steps. Although rationales do not explicitly specify programs, they provide a scaffolding for their structure via intermediate milestones. To evaluate our approach, we have created a new 100,000-sample dataset of questions, answers and rationales. Experimental results show that indirect supervision of program learning via answer rationales is a promising strategy for inducing arithmetic programs.",}
```
 
* CommonSense QA
  
```
@inproceedings{talmor-etal-2019-commonsenseqa,
    title = "{C}ommonsense{QA}: A Question Answering Challenge Targeting Commonsense Knowledge",
    author = "Talmor, Alon  and
      Herzig, Jonathan  and
      Lourie, Nicholas  and
      Berant, Jonathan",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1421",
    doi = "10.18653/v1/N19-1421",
    pages = "4149--4158",
    abstract = "When answering a question, people often draw upon their rich world knowledge in addition to the particular context. Recent work has focused primarily on answering questions given some relevant document or context, and required very little general background. To investigate question answering with prior knowledge, we present CommonsenseQA: a challenging new dataset for commonsense question answering. To capture common sense beyond associations, we extract from ConceptNet (Speer et al., 2017) multiple target concepts that have the same semantic relation to a single source concept. Crowd-workers are asked to author multiple-choice questions that mention the source concept and discriminate in turn between each of the target concepts. This encourages workers to create questions with complex semantics that often require prior knowledge. We create 12,247 questions through this procedure and demonstrate the difficulty of our task with a large number of strong baselines. Our best baseline is based on BERT-large (Devlin et al., 2018) and obtains 56{\%} accuracy, well below human performance, which is 89{\%}.",
}
```
* StrategyQA
  
```
@article{geva-etal-2021-aristotle,
    title = "Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies",
    author = "Geva, Mor  and
      Khashabi, Daniel  and
      Segal, Elad  and
      Khot, Tushar  and
      Roth, Dan  and
      Berant, Jonathan",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "9",
    year = "2021",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2021.tacl-1.21",
    doi = "10.1162/tacl_a_00370",
    pages = "346--361",
    abstract = "A key limitation in current datasets for multi-hop reasoning is that the required steps for answering the question are mentioned in it explicitly. In this work, we introduce StrategyQA, a question answering (QA) benchmark where the required reasoning steps are implicit in the question, and should be inferred using a strategy. A fundamental challenge in this setup is how to elicit such creative questions from crowdsourcing workers, while covering a broad range of potential strategies. We propose a data collection procedure that combines term-based priming to inspire annotators, careful control over the annotator population, and adversarial filtering for eliminating reasoning shortcuts. Moreover, we annotate each question with (1) a decomposition into reasoning steps for answering it, and (2) Wikipedia paragraphs that contain the answers to each step. Overall, StrategyQA includes 2,780 examples, each consisting of a strategy question, its decomposition, and evidence paragraphs. Analysis shows that questions in StrategyQA are short, topic-diverse, and cover a wide range of strategies. Empirically, we show that humans perform well (87{\%}) on this task, while our best baseline reaches an accuracy of ∼ 66{\%}.",
}
```
### Requirements
* See ```requirements.txt``` (the main big 3 libraries are Huggingface Transformers, vLLM, and PyTorch - the rest mostly being dependencies.)
* [vLLM](https://vllm.readthedocs.io/en/latest/) needs to be built from the source at the moment. Use this [branch](https://github.com/vllm-project/vllm/tree/6fc2a38b110f9ba6037b31ee016f20df32426877) for consistency but the latest version would probably work too.

### Model Setup

The code base is mainly set up to work with vLLM-compatible models. 

* A few models like [LLAMA-instruct](https://huggingface.co/upstage/llama-30b-instruct-2048) and [Redmond](https://huggingface.co/NousResearch/Redmond-Puffin-13B) are already setup. But change the model weight paths as you need in ```model.py``` (see the constructor of generator class). I use locally downloaded checkpoints paths so it will not work out of the box unless you download the checkpoints in a similar path or change the path.

If you want to add a new model (vLLM compatible) do the following:

1. Add the model name in ```argparser.py``` for the ```model``` option.
2. Add a prompt template for that specific model name in ```prompt.py``` (see examples in the end of the code file) (optional; there is a default prompt but probably wouldn't be optimal).  
3. Associate the model name (the one you defined in the argparser.py) with a checkpoint path in ```model.py``` (see the constructor of generator class in that file for examples).

### Run

This a general template for code execution:
```
python main.py --search_style=MultiSearch --model=LLAMA30_instruct --gpu_ids="0,1" --prompt_style=cot --dataset=gsm8k --reward_types=confidence+correctness
```
This would run [LLAMA 30B instruct](https://huggingface.co/upstage/llama-30b-instruct-2048) with zero-shot chain-of-thought (```cot```) prompt using ```MultiSearch``` (to be explained below) as a search strategy on ```GSM8K``` using a reward function (```confidence+correctness``` - to be explained below) for search guidance. The model weights will be distributed to cuda:0 and cuda:1 (given ```gpu_ids="0,1"```).

Some other salient arguments:
* ```checkpoint``` - set it true if you are loading some earlier saved checkpoints (checkpoints are automatically saved)
* ```SC``` - set it true to enable self-consistency [1]. Only relevant if ```search_style=none```. 

The available options for each argument and the defaults can be found in ``argparser.py``.

### Logs

You will find execution logs in ```logs/```

### Prompt Styles

In this project, we employ various search strategies at the level of reasoning steps (each reasoning step counts as "a single move in the game"). 
This also raises the question of how to decompose the generation into a series of steps. One way to do that is to create any arbitrary prompt template with a clear structure (that can be used for parsing steps) and then use a few shot examples with the template to prime the model to follow the structure. Here, however, I am interested in the zero-shot regime. I try to use zero-shot prompt instructions in specific ways to incite different forms of auto-decomposition of steps. Below, I discuss all the prompt styles used in this project and their corresponding decomposition methodology.

1. **Chain-of-Thoughts** (```cot```) - This uses the standard zero-shot COT prompt ```Let's think step by step.``` [2]. For step decomposition ```\n``` (new line) is used. There are some additional setups for properly ignoring empty new lines and such. Ultimately, this isn't necessarily an ideal way to decompose reasoning steps, not all new lines in the COT results would be complete reasoning steps but it's a baseline starting point that can be done in a zero-shot manner.

2. **Chain-of-Thoughts Step** (```cot_step```) - This is a simple extension of zero-shot COT: ```Let's think step by step. Step 1: ```. This automatically primes the language model to organize its chain of reasoning in numbered steps (Step1: xyz Step2: abc ...). This structure can be then easily used for decomposing steps.

3. **PS+** (```ps```) - This is the zero-shot plan and solve prompt (PS+ version) introduced in [3]. New line decomposition is used similar to ```cot```.

4. **Tabular Chain-of-Thoughts** (```cot_tab```) - This is the zero-shot tabular cot prompt introduced in [4] - ```\n|step|subquestion|process|result|\n```. It's a simple way to produce a structured tabular format reasoning steps. We use newline for decomposition again but unlike before newline decomposition is more meaningful here - because each decomposed newline will correspond to a step in the table. 

5. **Struct** (```struct```) - This prompt uses elements of many of the above prompts. It provides detailed instructions to decompose the solution/answer into steps and substeps (with subproblem identification, relevant facts, and solution). This produces highly structured results and can be decomposed according to the structure similar to ```cot```. Details of the prompt can be found in ```prompt.py``` and the decomposition code can be found in ```node_transition.py```.

6. **Struct Minimal** (```struct_min```) - It's similar to struct with one less substep. Details in ```prompt.py```. I haven't run this variant - there could be bugs. 

You can modify ```prompt.py``` to add a few shot prompts. 

### Search Styles

All the search codes can be found in ```Search/```.

* ```none``` - This method doesn't apply any particular search strategy besides standard autoregressive greedy decoding. This can be combined with ```SC=True``` for self-consistency with multiple sampling.
* ```MultiSearch``` - This strategy uses multi-sampling. Then the rewards for each sample (accumulative reward for each decomposed step) are generated after the fact. Rewards are used for voting answers in various ways to be described later.
* ```MultiGreedy``` - This strategy uses greedy search but at the level of steps (unlike ```none```). At every iteration, given the history chain of reasoning steps, the model generates some k next reasoning step candidates. Each of the k candidates are then scored (assigned a reward). Then the maximum-scoring candidate is selected. This strategy is used in parallel for multiple samples of initial reasoning steps which leads to multiple search results which can be used for self-consistency. 
  This search strategy can be thought of as similar to DFS from Tree of Thought [5] but without any backtracking.
* ```BeamSearch``` - This is the beam search version of the above. The implementation is inspired by [6]. Moreover, this method can be thought of as similar to the BFS method (with truncated frontiers) used in Tree of Thought [5].
* ```DivBeamSearch``` - This is the same as beam search but encourages more diversity in generation by restricting siblings. In each iteration a maximum of m (m << beam size (I use m=2)) siblings are allowed. If there is space in the beam size after choosing all candidates following this restricting the residual candidates are added based on their rewards. The idea is similar in spirit to [7] but we don't strictly modify the equation of the scoring with a penalty - but use more of a hard constraint as described.
* ```MCTS``` - This is Monte Carlo Tree Search. The implementation follows the structure [here](https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/) roughly. It takes some inspiration from [8].
* ```SPMCTS``` - This implementation (Semi-Parallel Monte Carlo Tree Search) parallelizes MCTS a bit more. It selects multiple leaves at once and rolls out multiple paths at once. As such, it takes fewer sequential iterations. The end result is a similar number of samples as MCTS. The implementations are not computationally equivalent however and not intended to be.

Note while some of the methods are inspired by prior work, they are not attempted to be perfectly faithful implementations of those papers. 


### Reward Types

Similar to [6,8] the reward for each step is calculated based on self-evaluation and confidence of generation of the step (based on logprobs). 
The self-evaluation technique uses LLMs to evaluate its own generations by asking multi-choice questions (MCQ) about the helpfulness/correctness of the step (the self-evaluation questions can be found in ```rewards.py```).

Different types of reward types and combos are available as arguments in ```argparse.py```.

* ```confidence``` - Only uses confidence (based on logprobs) as the reward.
* ```correctness``` - Only uses answer probabilities from a step correctness-related MCQ as the reward.
* ```helpfulness``` - Only uses answer probabilities from a step helpfulness-related MCQ as the reward
* ```both``` - Uses both ```correctness``` and ```helpfulness```.
* ```confidence+correctness``` - Uses both ```confidence``` and ```correctness```.
* ```confidence+helpfulness``` - Uses both ```confidence``` and ```helpfulness```.
* ```confidence+both``` - Uses all of ```confidence```, ```correctness```, and ```helpfulness```.

NB: Prompt styles ```struct``` and ```struct-min``` follow different rules. They have fine-grained substep structures and use rewards appropriate for those. The subproblem identification substep only uses helpfulness-related reward (because it technically isn't a reasoning step to be correct/incorrect) and the solution step only uses the correctness-related question (because presumably, the subproblem helpfulness will correlate with the helpfulness of the solution).

The MCQs used for different prompts and different reward types for self-evaluation can be found in ```reward.py```.

### Answer Voting

Several types of answer voting mechanisms are implemented and automatically tracked simultaneously in ```main.py``` (not related to any ```argparse.py``` options). They are:

1. **Majority Voting** (```Voted Answer``` in logs) - Just simple majority voting [1].
2. **Reward Voting** (```Reward Voted Answer``` in logs) - Similar to majority voting, but the value of each vote is the reward ($\in [0,1]$) of the corresponding answer path rather than just 1 for all,
3. **Top K Reward Voting** (```Top K Reward Voted Answer``` in logs) - Select Top K (we use K=5) highest rewarded answers then apply reward voting among them. This allows filtering potentially "bad" low-reward answers which can potentially add up to votes.
4. **Max Reward** (```Max Reward Answer``` in logs) - Select the answer with the maximum reward.

### Limitations

* **Bugs** - The project is more of an experimental prototype. There can be some bugs in the more sophisticated search strategies. They do run, but there can be some implementation issues that I may need to double-check. Personally, I didn't have much luck with the more sophisticated methods in some toy runs. Generally, just none or none+SC works pretty well with hard-to-beat results.
* **Batch** - There isn't any batching in the code. It executes one sample at a time. There is some internal batching in SPMCTS, BeamSearch, or such for parallelly running rewards in multiple children, or getting child candidates for multiple parent nodes in the beam. But all that is related to a single question/prompt. I also tried to parallelize some aspects of ```node_transition.py``` a bit more (particularly the reward computation) but didn't get much benefit in empirical time cost.
* **Cache** - One limitation that bottlenecks the performance of these models is presumably the lack of reuse of Key-value caching. To get rewards, I have to generally terminate after a reasoning step is calculated. This resets the cache for future generations. Moreover, reward generation (self-evaluation) requires rebuilding the cache. All of these require rebuilding the KV cache multiple times. Better cache reuse may significantly speed up the search strategies besides ```none``` (which is currently the fastest). But this may require modifying vLLM and huggingface Transformers.
* **Documentation** - I need to add more documentation (here or in a paper). But in the meantime, for any questions or anything else contact the email (linked to my GitHub account).
* **Misc** - It also goes without saying there are other endless things that can be added like complexity-based weighing, automatically retrieving examples with synthetic ground truth (bootstrapping) for few shot prompts, or multi-agent debate to name a few. 

### Related Works

[1] Self-Consistency Improves Chain of Thought Reasoning in Language Models 

```
@inproceedings{
wang2023selfconsistency,
title={Self-Consistency Improves Chain of Thought Reasoning in Language Models},
author={Xuezhi Wang and Jason Wei and Dale Schuurmans and Quoc V Le and Ed H. Chi and Sharan Narang and Aakanksha Chowdhery and Denny Zhou},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=1PL1NIMMrw}
}
```

[2] Large Language Models are Zero-Shot Reasoners

```
@inproceedings{NEURIPS2022_8bb0d291,
 author = {Kojima, Takeshi and Gu, Shixiang (Shane) and Reid, Machel and Matsuo, Yutaka and Iwasawa, Yusuke},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {22199--22213},
 publisher = {Curran Associates, Inc.},
 title = {Large Language Models are Zero-Shot Reasoners},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/8bb0d291acd4acf06ef112099c16f326-Paper-Conference.pdf},
 volume = {35},
 year = {2022}
}

```

[3] Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models

```
@inproceedings{wang-etal-2023-plan,
    title = "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models",
    author = "Wang, Lei  and
      Xu, Wanyu  and
      Lan, Yihuai  and
      Hu, Zhiqiang  and
      Lan, Yunshi  and
      Lee, Roy Ka-Wei  and
      Lim, Ee-Peng",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.147",
    pages = "2609--2634",
    abstract = "Large language models (LLMs) have recently been shown to deliver impressive performance in various NLP tasks. To tackle multi-step reasoning tasks, Few-shot chain-of-thought (CoT) prompting includes a few manually crafted step-by-step reasoning demonstrations which enable LLMs to explicitly generate reasoning steps and improve their reasoning task accuracy. To eliminate the manual efforts, Zero-shot-CoT concatenates the target problem statement with {``}\textit{Let{'}s think step by step}{''} as an input prompt to LLMs. Despite the success of Zero-shot-CoT, it still suffers from three pitfalls: calculation errors, missing-step errors, and semantic misunderstanding errors. To address the missing-step errors, we propose Plan-and-Solve (PS) Prompting. It consists of two components: first, devising a plan to divide the entire task into smaller subtasks, and then carrying out the subtasks according to the plan. To address the calculation errors and improve the quality of generated reasoning steps, we extend PS prompting with more detailed instructions and derive PS+ prompting. We evaluate our proposed prompting strategy on ten datasets across three reasoning problems. The experimental results over GPT-3 show that our proposed zero-shot prompting consistently outperforms Zero-shot-CoT across all datasets by a large margin, is comparable to or exceeds Zero-shot-Program-of-Thought Prompting, and has comparable performance with 8-shot CoT prompting on the math reasoning problem. The code can be found at https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting.",
}
```

[4] Tab-CoT: Zero-shot Tabular Chain of Thought

```
@inproceedings{ziqi-lu-2023-tab,
    title = "Tab-{C}o{T}: Zero-shot Tabular Chain of Thought",
    author = "Ziqi, Jin  and
      Lu, Wei",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.651",
    pages = "10259--10277",
    abstract = "The chain-of-though (CoT) prompting methods were successful in various natural language processing (NLP) tasks thanks to their ability to unveil the underlying complex reasoning processes.Such reasoning processes typically exhibit highly structured steps.Recent efforts also started investigating methods to encourage more structured reasoning procedures to be captured (cite least to most).In this work, we propose Tab-CoT, a novel tabular-format CoT prompting method, which allows the complex reasoning process to be explicitly modeled in a highly structured manner.Despite its simplicity, we show that our approach is capable of performing reasoning across multiple dimensions (i.e., both rows and columns).We demonstrate our approach{'}s strong zero-shot and few-shot capabilities through extensive experiments on a range of reasoning tasks.",
}
```

[5] Tree of Thoughts: Deliberate Problem Solving with Large Language Models

```
@misc{yao2023tree,
      title={Tree of Thoughts: Deliberate Problem Solving with Large Language Models}, 
      author={Shunyu Yao and Dian Yu and Jeffrey Zhao and Izhak Shafran and Thomas L. Griffiths and Yuan Cao and Karthik Narasimhan},
      year={2023},
      eprint={2305.10601},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

[6]  Decomposition Enhances Reasoning via Self-Evaluation Guided Decoding

```
@misc{xie2023decomposition,
      title={Decomposition Enhances Reasoning via Self-Evaluation Guided Decoding}, 
      author={Yuxi Xie and Kenji Kawaguchi and Yiran Zhao and Xu Zhao and Min-Yen Kan and Junxian He and Qizhe Xie},
      year={2023},
      eprint={2305.00633},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

[7] A Simple, Fast Diverse Decoding Algorithm for Neural Generation

```
@misc{li2016simple,
      title={A Simple, Fast Diverse Decoding Algorithm for Neural Generation}, 
      author={Jiwei Li and Will Monroe and Dan Jurafsky},
      year={2016},
      eprint={1611.08562},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

[8] Reasoning with Language Model is Planning with World Model

```
@misc{hao2023reasoning,
      title={Reasoning with Language Model is Planning with World Model}, 
      author={Shibo Hao and Yi Gu and Haodi Ma and Joshua Jiahua Hong and Zhen Wang and Daisy Zhe Wang and Zhiting Hu},
      year={2023},
      eprint={2305.14992},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


