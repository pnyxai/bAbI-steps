import json
import random
import shutil
from pathlib import Path

import numpy as np

from babisteps import logger
from babisteps import utils as ut

# Global constants
SPLITS = ["test", "train", "validation"]


def prepare_path(path: Path,
                 folder_name: str,
                 logger=None,
                 delete_if_exists: bool = False):
    # Define the folder path
    folder_path = path / folder_name  # Assuming `path` is a Path object
    # Check if the folder exists
    if folder_path.exists() and delete_if_exists:
        if logger:  # Only log if logger is provided
            logger.info("Clearing content", folder=folder_path)

        # Remove all contents inside the folder
        for item in folder_path.iterdir():
            if item.is_file() or item.is_symlink():
                if logger:  # Only log if logger is provided
                    logger.info("Deleting file", file=item)
                item.unlink()  # Remove file or symlink
            elif item.is_dir():
                if logger:  # Only log if logger is provided
                    logger.info("Deleting folder and its contents",
                                folder=item)
                shutil.rmtree(item)  # Remove directory and its contents
    else:
        if logger:  # Only log if logger is provided
            logger.info("Folder does not exist. Creating it.",
                        folder=folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
    if logger:  # Only log if logger is provided
        logger.info("Folder is now ready for use.", folder=folder_path)
    return folder_path


def save_as_jsonl(json_list: list[dict],
                  folder_path: Path,
                  logger,
                  filename="output.jsonl"):
    """
    Saves a list of JSON objects as a JSONL file inside the specified folder.

    :param json_list: List of dictionaries to save
    :param folder_path: Path object representing the target folder
    :param filename: Name of the JSONL file (default: output.jsonl)
    """
    # shuffle the json_list
    import random
    random.shuffle(json_list)

    try:
        file_path = folder_path / filename  # Construct full file path

        with open(file_path, "w", encoding="utf-8") as f:
            for obj in json_list:
                f.write(json.dumps(obj) +
                        "\n")  # Write each JSON object on a new line
        if logger:
            logger.info("Saved JSONL",
                        file_path=file_path)  # Print confirmation
    except Exception as e:
        if logger:
            logger.error("Error saving JSONL", error=str(e))
        raise e
    return file_path


def save_as_txt(text: str, folder_path: Path, logger, filename="output.txt"):
    """
    Saves a txt plain text.

    :param txt: Text to save
    :param folder_path: Path object representing the target folder
    :param filename: Name of the .txt file (default: output.txt)
    """
    try:
        file_path = folder_path / filename  # Construct full file path
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        if logger:
            logger.info("Saved .txt",
                        file_path=file_path)  # Print confirmation
    except Exception as e:
        if logger:
            logger.error("Error saving .txt", error=str(e))
        raise e


def _run_generation(g, yaml_cfg):
    retries = 0
    while retries < yaml_cfg["max_retries"]:
        try:
            g.generate()
            break
        except Exception as _:
            g = g.recreate()
            retries += 1
    if retries >= yaml_cfg["max_retries"]:
        raise Exception(
            "Max retries reached from the same seeded generator. Skipping story"
        )

    json = g.get_json()
    txt = g.get_txt()
    g.close_logger()
    return json, txt


def process_single_task(task_name_i,
                        task_path_i,
                        yaml_cfg,
                        log_queue,
                        seed_offset,
                        optimistic_generation=True):
    """Process a single task - designed to run in a separate process.
    
    Args:
        task_name_i: Name of the task to process
        task_path_i: Path to the task configuration
        yaml_cfg: Configuration dictionary
        log_queue: Queue for logging messages
        seed_offset: Offset to add to random seeds for this process
        optimistic_generation: If True, skip failed samples. If False, raise exception on failure.
    """
    task_logger = logger.QueueLogger(log_queue, task_name_i)

    # Set seeds with offset for this process
    if yaml_cfg.get('seed') is not None:
        random.seed(yaml_cfg['seed'] + seed_offset)
    if yaml_cfg.get('numpy_random_seed') is not None:
        np.random.seed(yaml_cfg['numpy_random_seed'] + seed_offset)

    task_logger.info("Running task %s", task_name_i)
    func_i = ut.load_function_from_config(task_path_i)

    jsonl_file_paths = {}
    max_generator_rebuildings = yaml_cfg.get("max_generator_rebuildings")

    # Process each split
    for split in SPLITS:
        jsonl_dataset = []
        txt_dataset = ""
        framework, generator_func, folder_path = func_i(**yaml_cfg)

        if split == "test":
            task_logger.info("STARTING NEW TASK:",
                             task=task_name_i,
                             split=split)
        idx = 0
        list_to_retry = []
        # Iterate over all leaves
        for leaf, answer, count in framework:
            if split == "test":
                task_logger.info("Starting new leaf's generation",
                                 task=task_name_i,
                                 leaf=leaf.__name__,
                                 answer=answer,
                                 count=count)
            # Generate samples for this leaf
            for i in range(count):
                if idx == 50 and split != "test":
                    break
                # get generator
                g = generator_func(leaf, answer, i)
                generations_tries = 0
                story_completed = False
                # Try generating the story with retries
                while generations_tries < max_generator_rebuildings:
                    try:
                        json_i, txt_i = _run_generation(g, yaml_cfg)
                        story_completed = True
                        break
                    except Exception as e:
                        generations_tries += 1
                        task_logger.debug(str(e),
                                          task=task_name_i,
                                          leaf=leaf.__name__,
                                          answer=answer,
                                          idx=idx,
                                          i=i)

                        if generations_tries == max_generator_rebuildings:
                            json_i, txt_i = None, None

                            if not optimistic_generation:
                                # Non-optimistic mode: fail the entire process
                                task_logger.error(
                                    "Max tries reached even for differents seeded generator. Failing process.",
                                    task=task_name_i,
                                    leaf=leaf.__name__,
                                    answer=answer,
                                    idx=idx,
                                    i=i)
                                error_details = (
                                    f"Generation failed for task '{task_name_i}', "
                                    f"leaf '{leaf.__name__}', answer '{answer}', "
                                    f"idx={idx}, i={i}. Max retries ({max_generator_rebuildings}) reached."
                                )
                                raise RuntimeError(error_details) from None
                            else:
                                # Optimistic mode: skip and continue
                                task_logger.error(
                                    "Max tries reached even for differents seeded generator. Skipping sample",
                                    task=task_name_i,
                                    leaf=leaf.__name__,
                                    answer=answer,
                                    idx=idx,
                                    i=i)
                            break

                if story_completed is False:
                    element_to_retry = (split, leaf, answer, idx, i)
                    list_to_retry.append(element_to_retry)
                else:
                    json_i['idx'] = idx
                    jsonl_dataset.append(json_i)
                    txt_dataset += txt_i
                idx += 1

                if split == "test" and idx % (
                        yaml_cfg["num_samples_by_task"] // 10) == 0:
                    percentage = (idx / yaml_cfg["num_samples_by_task"]) * 100
                    percentage = "{:.0f}%".format(percentage)
                    task_logger.info("Progress",
                                     task=task_name_i,
                                     percentage=percentage)

        try:
            jsonl_file_path_i = save_as_jsonl(jsonl_dataset,
                                              folder_path,
                                              None,
                                              filename=f"{split}.jsonl")
            jsonl_file_paths[split] = jsonl_file_path_i
            save_as_txt(txt_dataset,
                        folder_path,
                        None,
                        filename=f"{split}.txt")
        except Exception as e:
            task_logger.error("Error saving dataset",
                              task=task_name_i,
                              error=str(e))
            raise Exception("Error saving dataset") from e

        if split == "test":
            task_logger.info("SUCCESS Generation", task=task_name_i)

    return task_name_i, jsonl_file_paths.get(
        "test")  # Return for dataset creation
