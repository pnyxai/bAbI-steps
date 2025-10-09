import argparse
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from babisteps import datasets as ds
from babisteps import logger
from babisteps import proccesing as proc
from babisteps import utils as ut


def split_or_empty(value):
    return value.split(",") if value else []


def replace_path_in_file(file_path, new_path, logger):
    """Reads a file, replaces the placeholder, and writes the changes back."""
    try:
        with open(file_path) as f:
            content = f.read()
        # Conver new path to absolute path
        abs_new_path = os.path.abspath(new_path)
        modified_content = content.replace("<PATH/TO/REPLACE>", abs_new_path)
        with open(file_path, 'w') as f:
            f.write(modified_content)
        logger.info("Placeholder replaced", file_path=file_path)
        return True
    except FileNotFoundError:
        logger.error("File not found at path", file_path=file_path)
        return False


def main():
    parser = argparse.ArgumentParser(description="Create a dataset of stories")
    parser.add_argument(
        "--task_path",
        type=str,
        default="./babisteps/tasks",
        help="Root path where are placed the tasks",
    )
    parser.add_argument(
        "--tasks",
        type=split_or_empty,
        default=["all"],
        help="List of the taks name to run the experiments",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs",
        help="Path to save the script results",
    )
    parser.add_argument(
        "--include_lm_eval_template",
        default=True,
        help="Include the lm_eval template in the output folder",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for python's random module. Default set to 0.",
    )
    parser.add_argument(
        "--numpy_random_seed",
        type=int,
        default=1234,
        help="Random seed for numpy's random module. Default set to 1234.",
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level.",
    )
    args = parser.parse_args()

    # Get base output folder path
    output_path = Path(args.output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # Add date string
    current_date = datetime.now()
    output_path = Path(
        os.path.join(output_path, current_date.strftime("%Y-%m-%dT%H:%M:%S")))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    dataset_path = Path(os.path.join(output_path, "babisteps"))
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    jsonl_path = Path(os.path.join(output_path, "jsonl"))
    if not os.path.exists(jsonl_path):
        os.mkdir(jsonl_path)
    # init of jsonl_path
    start_time = time.time()
    main_logger = logger.get_logger(
        "main",
        level=getattr(logging, f"{args.verbosity}"),
        log_file=os.path.join(output_path, "logs.txt"),
    )
    task_path = Path(args.task_path)
    if not os.path.exists(task_path):
        raise ValueError(f"Path {task_path} does not exist")
    # get task_folders as the folder inside the task_path (each element as Path)
    task_folders = [f for f in task_path.iterdir() if f.is_dir()]
    common_yaml = task_path / "commons.yaml"
    if not os.path.exists(common_yaml):
        raise ValueError(f"File {common_yaml} does not exist")
    # load common.yaml
    with open(common_yaml) as stream:
        try:
            yaml_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise ValueError(f"Error loading {common_yaml}") from exc

    # update yaml_cfg with output_path
    yaml_cfg["output_path"] = jsonl_path
    task_path_dict = ut.create_task_path_dict(args.tasks, task_folders,
                                              main_logger)

    seed_message = []
    if args.seed is not None:
        # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
        seed_message.append(f"Setting random seed to {args.seed}")
        random.seed(args.seed)

    if args.numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {args.numpy_random_seed}")
        np.random.seed(args.numpy_random_seed)
    if seed_message:
        main_logger.info(" | ".join(seed_message))

    jsonl_path_dict = {}
    max_generator_rebuildings = yaml_cfg.get("max_generator_rebuildings")
    for task_name_i, task_path_i in task_path_dict.items():
        main_logger.info("Running task %s", task_name_i)
        splits = ["test", "train", "validation"]
        func_i = ut.load_function_from_config(task_path_i)
        for split in splits:
            jsonl_dataset = []
            txt_dataset = ""
            framework, generator_func, folder_path = func_i(**yaml_cfg)
            if split == "test":
                main_logger.info("STARTING NEW TASK:",
                                task=task_name_i,
                                split=split)
            idx = 0
            list_to_retry = []
            for leaf, answer, count in framework:
                if split == "test":
                    main_logger.info("Starting new leaf's generation",
                                    task=task_name_i,
                                    leaf=leaf.__name__,
                                    answer=answer,
                                    count=count)
                for i in range(count):
                    if idx == 50 and split != "test":
                        break
                    g = generator_func(leaf, answer, i)
                    generations_tries = 0
                    story_completed = False
                    while generations_tries < max_generator_rebuildings:
                        try:
                            json_i, txt_i = proc._run_generation(g, yaml_cfg)
                            story_completed = True
                            break
                        except Exception as e:
                            generations_tries += 1
                            main_logger.debug(
                                e,
                                task=task_name_i,
                                leaf=leaf,
                                answer=answer,
                                idx=idx,
                                i=i
                            )

                            if generations_tries == max_generator_rebuildings:
                                json_i, txt_i = None, None
                                main_logger.error(
                                    "Max tries reached even for differents seeded generator. Skipping sample",
                                    task=task_name_i,
                                    leaf=leaf,
                                    answer=answer,
                                    idx=idx,
                                    i=i
                                )
                                break

                    if story_completed is False:
                        element_to_retry = (split, leaf, answer, idx, i)
                        list_to_retry.append(element_to_retry)
                    else:
                        json_i['idx'] = idx
                        jsonl_dataset.append(json_i)
                        txt_dataset += txt_i
                    idx += 1
                    
                    if split == "test" and idx % (yaml_cfg["num_samples_by_task"] // 10) == 0:
                        percentage = (idx / yaml_cfg["num_samples_by_task"]) * 100
                        # percentage to str with 0 decimals
                        percentage = "{:.0f}%".format(percentage)
                        main_logger.info("Progress", task=task_name_i, percentage=percentage)


            try:
                jsonl_file_path_i = proc.save_as_jsonl(
                    jsonl_dataset,
                    folder_path,
                    None,
                    filename=f"{split}.jsonl")
                jsonl_path_dict[task_name_i] = jsonl_file_path_i
                proc.save_as_txt(txt_dataset,
                                 folder_path,
                                 None,
                                 filename=f"{split}.txt")
            except Exception as e:
                raise Exception("Error saving dataset") from e
            if split == "test":
                main_logger.info("SUCCESS Generation", task=task_name_i)

    main_logger.info("STARTING DATASET CREATION")
    ds.create_babisteps_dataset(dataset_path, jsonl_path_dict, main_logger,
                                splits)
    main_logger.info("SUCCESS DATASET CREATION")

    # LM-EVAL-HARNESS TEMPLATE CREATION
    if args.include_lm_eval_template:
        main_logger.info("STARTING LM-EVAL-HARNESS TEMPLATE CREATION")
        src_path = Path(__file__).parent / "babisteps" / "lm_eval"
        lm_eval_path = Path(os.path.join(output_path, "lm_eval"))
        if not os.path.exists(lm_eval_path):
            os.mkdir(lm_eval_path)
        if not os.path.exists(src_path):
            raise ValueError(f"Path {src_path} does not exist")
        # copy all the content of src_path to lm_eval_path usingh shutil
        import shutil
        shutil.copytree(src_path, lm_eval_path, dirs_exist_ok=True)
        # get the path of files named '_default_template_yaml'
        # in the lm_eval_path
        df_yamls = [
            file
            for file in Path(lm_eval_path).rglob("*_default_template_yaml")
        ]
        for file in df_yamls:
            # replace the path in the file with the new path
            if not replace_path_in_file(file, str(dataset_path), main_logger):
                main_logger.warning("File not found", file=file)

    end_time = time.time()
    main_logger.info("Execution time", time=end_time - start_time)


if __name__ == "__main__":
    main()
