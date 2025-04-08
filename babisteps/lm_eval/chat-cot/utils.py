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
        prompt += ".{}\n".format(opt)
    return prompt


doc_to_text = partial(format_example, including_answer=False)
fewshot_to_text = partial(format_example, including_answer=True)


@register_filter("GetResponse")
class GetResponse(Filter):
    """ """

    def apply(self, resps, docs):
        filtered_resps = []

        for r, doc in zip(resps, docs):
            filtered = []
            for resp in r:
                if "</think>" in resp:
                    # Remove CoT content
                    resp = resp.split("</think>")[-1]
                else:
                    # Filter everything after the first break line
                    # regex_pattern: "^(.*?)(?=\\n|$)"
                    match = re.search(r"^(.*?)(?=\n|$)", resp)
                    if match:
                        resp = match.group(1)
                # Remove leading white spaces
                resp = resp.lstrip()
                # function to ignore right white spaces or line breaks
                # regex_pattern: "^(.*?)\\s*$"
                match_trailing = re.search(r"^(.*?)\s*$", resp)
                if match_trailing:
                    resp = match_trailing.group(1)
                filtered.append(resp)
            filtered_resps.append(filtered)

        return filtered_resps
