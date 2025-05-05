"""
Take in a YAML, and output all "other" splits with this YAML
"""

import argparse
import os

import yaml
from tqdm import tqdm

from babisteps.datasets import TASKS2NAME

# TO BE USED LATER WITH CUSTOM LM-EVAL-HARNESS FUNCTIONS
IMMEDIATEORDER_CFG = """\
fewshot_config:
  sampler: default
  doc_to_text: !function utils.rnd_choice_fewshot_to_text
  doc_to_target: ""
doc_to_text: !function utils.rnd_choice_doc_to_text
"""

CHAT_IMMEDIATEORDER_CFG = """\
fewshot_config:
  sampler: default
  doc_to_text: !function utils.fewshot_to_text
  doc_to_target: "{{answer | random}}"
"""

LISTING_CFG = """\
fewshot_config:
  sampler: default
  doc_to_text: !function utils.listing_fewshot_to_text
  doc_to_target: ""
doc_to_text: !function utils.listing_doc_to_text
process_results: !function utils.process_results_listing
filter_list:
  - name: get_response
    filter:
      - function: "lowercase"    
      # Filter everything after the first break line, ignoring leading newlines
      - function: "regex"
        regex_pattern: "^[\\n]*([^\\n]*)" # Updated regex
      # Remove leading white spaces
      - function: remove_whitespace
      # function to ignore right white spaces or line breaks
      - function: "regex"
        regex_pattern: "^(.*?)\\\\s*$"
      - function: "replace_regex" # Use the new filter type      
        regex_pattern: " and" # The literal string " and" to match
        replacement_string: "" # Replace with an empty string (this is the default)
      - function: take_first
"""

CHAT_LISTING_CFG = """\
fewshot_config:
  sampler: default
  doc_to_text: !function utils.fewshot_to_text
  doc_to_target: "{{answer | join(', ')}}"
process_results: !function utils.process_results_listing
filter_list:
  - name: get_response
    filter:
      - function: "lowercase"    
      # Filter everything after the first break line, ignoring leading newlines
      - function: "regex"
        regex_pattern: "^[\\n]*([^\\n]*)" # Updated regex
      # Remove leading white spaces
      - function: remove_whitespace
      # function to ignore right white spaces or line breaks
      - function: "regex"
        regex_pattern: "^(.*?)\\\\s*$"
      - function: "replace_regex" # Use the new filter type      
        regex_pattern: " and" # The literal string " and" to match
        replacement_string: "" # Replace with an empty string (this is the default)
      - function: take_first
"""

DICT_CFG = {}
COT_DICT_CFG = {}
DICT_CFG["2"] = IMMEDIATEORDER_CFG
COT_DICT_CFG["2"] = CHAT_IMMEDIATEORDER_CFG
DICT_CFG["4"] = LISTING_CFG
COT_DICT_CFG["4"] = CHAT_LISTING_CFG
DICT_CFG["5"] = IMMEDIATEORDER_CFG
COT_DICT_CFG["5"] = CHAT_IMMEDIATEORDER_CFG
DICT_CFG["6"] = IMMEDIATEORDER_CFG
COT_DICT_CFG["6"] = CHAT_IMMEDIATEORDER_CFG
DICT_CFG["7"] = IMMEDIATEORDER_CFG
COT_DICT_CFG["7"] = CHAT_IMMEDIATEORDER_CFG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", required=True)
    parser.add_argument("--task_prefix", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our "other" YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    # with open(args.base_yaml_path, encoding="utf-8") as f:
    #     base_yaml = yaml.full_load(f)

    ALL_TASKS = []
    for task_id, task_name in tqdm(TASKS2NAME.items()):
        # split_name = f"task_{task_id}-{task_name}"
        task_name_use = f"task_{task_id}-{task_name}"
        if int(task_id) < 10:
            # To keep order correctly on display screen
            task_name_use = f"task_0{task_id}-{task_name}"
        if task_name_use not in ALL_TASKS:
            ALL_TASKS.append(task_name_use)

        description = (
            f"The following are basic taks (with answers) on the ability: {task_name}."
        )
        if args.task_prefix != "":
            description += ("\nReturn the answer to the question, without any "
                            "explanation.")

        yaml_dict = {
            "include":
            base_yaml_name,
            "task":
            f"babisteps-{args.task_prefix}-{task_name_use}"
            if args.task_prefix != "" else f"babisteps-{task_name_use}",
            "task_alias":
            task_name_use.replace("_", " ").replace("-", " - "),
            "dataset_name":
            task_name,
            "description":
            description,
        }

        file_save_path = args.save_prefix_path + f"_{task_name_use}.yaml"
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                allow_unicode=True,
                # default_style='"',
            )
        # To be used later with custom LM-EVAL-HARNESS functions
        if task_id in DICT_CFG and task_id in COT_DICT_CFG:
            # if task_prefix != "" then use COT_DICT_CFG:
            tmp_cfg = (COT_DICT_CFG[task_id]
                       if args.task_prefix != "" else DICT_CFG[task_id])
            with open(file_save_path, "a", encoding="utf-8") as yaml_file:
                yaml_file.write(tmp_cfg)
            yaml_file.close()
    if args.task_prefix != "":
        # Add
        babi_subcategories = [
            f"babisteps-{args.task_prefix}-{task}" for task in ALL_TASKS
        ]
    else:
        babi_subcategories = [f"babisteps-{task}" for task in ALL_TASKS]

    file_save_path = args.save_prefix_path + ".yaml"

    # eval_logger.info(f"Saving benchmark config to {file_save_path}")
    with open(file_save_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            {
                "group":
                f"babisteps-{args.task_prefix}-all"
                if args.task_prefix != "" else "babisteps-all",
                "task":
                babi_subcategories,
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
        )
