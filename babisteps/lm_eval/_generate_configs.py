"""
Take in a YAML, and output all "other" splits with this YAML
"""

import argparse
import logging
import os

import yaml
from tqdm import tqdm

#eval_logger = logging.getLogger("lm-eval")

from babisteps.datasets import TASKS2NAME


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
        #split_name = f"task_{task_id}-{task_name}"
        task_name_use = f"task_{task_id}-{task_name}"
        if int(task_id) < 10:
            # To keep order correctly on display screen
            task_name_use = f"task_0{task_id}-{task_name}"
        if task_name_use not in ALL_TASKS:
            ALL_TASKS.append(task_name_use)

        description = f"The following are basic taks on the subject of: {task_name}."

        yaml_dict = {
            "include": base_yaml_name,
            "task": f"babisteps-{args.task_prefix}-{task_name_use}"
            if args.task_prefix != ""
            else f"babisteps-{task_name_use}",
            "task_alias": task_name_use.replace("_", " ").replace("-", " - "),
            "dataset_name": task_name,
            "description": description,
        }

        file_save_path = args.save_prefix_path + f"_{task_name_use}.yaml"
        #eval_logger.info(f"Saving yaml for subset {task_name_use} to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                allow_unicode=True,
                #default_style='"',
            )
    if args.task_prefix != "":
        # Add
        babi_subcategories = [
            f"babisteps-{args.task_prefix}-{task}" for task in ALL_TASKS
        ]
    else:
        babi_subcategories = [f"babisteps-{task}" for task in ALL_TASKS]

    file_save_path = args.save_prefix_path + ".yaml"

    #eval_logger.info(f"Saving benchmark config to {file_save_path}")
    with open(file_save_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            {
                "group": f"babisteps-{args.task_prefix}-all"
                if args.task_prefix != ""
                else "babisteps-all",
                "task": babi_subcategories,
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
        )
