import re
from functools import partial

from lm_eval.api.registry import register_filter
from lm_eval.filters.extraction import Filter


def format_example(example, including_answer: bool):
    prompt = "\n\nWorld enumeration:\n"
    world = example["world_enumerate"]
    prompt += world
    prompt += "\n\nStory:\n"
    story = example["story"]
    prompt += story
    prompt += "\nQuestion:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(i, opt)
    prompt += "\nAnswer:\n"
    if including_answer:
        answer = example["answer"]
        prompt += answer
    return prompt


doc_to_text = partial(format_example, including_answer=False)
fewshot_to_text = partial(format_example, including_answer=True)