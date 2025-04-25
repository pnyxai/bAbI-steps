import importlib.util
import os
from pathlib import Path
from typing import Literal, get_args, get_origin

import numpy as np
import yaml


# from lm-eval-harnes argument parser
def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {
        kv[0]: handle_arg_string("=".join(kv[1:]))
        for kv in [arg.split("=") for arg in arg_list]
    }
    return args_dict


def _get_task_leaf_combinations(task_leaf_list):
    """
    Given a list of leaf/classes, iterate through each leaf/class and extract
    the possible values of the 'answer' Literal attribute.
    Returns a list of tuples (leaf, answer).
    """
    combinations = []
    for leaf in task_leaf_list:
        # Get the type of the answer from the class annotation.
        answer_type = leaf.__annotations__.get("answer")
        if answer_type is None:
            continue  # Skip if no answer type is defined.
        # Extract the possible values, handling nested Literal
        possible_answers = []
        try:
            # Get the arguments of the top-level type (e.g., Union)
            initial_args = get_args(answer_type)

            for arg in initial_args:
                if get_origin(arg) is Literal:
                    # If the argument is a Literal, get its values and add them
                    possible_answers.extend(get_args(arg))
                else:
                    # Otherwise (e.g., int, str, bool), add the type itself
                    possible_answers.append(arg)
        except Exception as e:
            # Handle cases where get_args might fail (e.g., annotation isn't a
            # type hint container )
            print(
                f"Warning: Could not process annotation for {leaf.__name__}: "
                f"{answer_type} - {e}")
            continue  # Skip this leaf

        # Now iterate through the processed possible values
        for ans in possible_answers:
            combinations.append((leaf, ans))

    return combinations


def generate_framework(num_samples: int, task_leaf_list: list):
    """
    Uses the list of classes to generate combinations (cls, answer)
    then distributes num_samples among them using a multinomial distribution.
    Returns a list of tuples (cls, answer, qty) where the total qty equals num_samples.
    """
    combinations = _get_task_leaf_combinations(task_leaf_list)
    n = len(combinations)
    if n == 0:
        raise ValueError("No valid class answer combinations found")
    probs = [1 / n] * n
    counts = np.random.multinomial(num_samples, probs)
    result = []
    for (leaf, answer), count in zip(combinations, counts):
        if count > 0:
            result.append((leaf, answer, int(count)))
    return result


def load_function_from_config(config_dir: Path):
    """
    Loads a function specified in a YAML config file.

    The config file should be in the given directory and named 'config.yaml'.
    In the same directory, there should be a 'utils.py' file.
    The config file must contain a 'func' key with the value in the format:
    "utils.<function_name>", where <function_name> is a function defined
    in the 'utils.py' file in the same directory.

    Args:
        config_dir (Path): Path to the directory containing config.yaml and utils.py.

    Returns:
        callable: The loaded function.

    Raises:
        FileNotFoundError: If the config.yaml or utils.py is not found in the directory.
        ValueError: If the 'func' key is missing or has an invalid format in config.yaml
                    or if the specified function is not found in utils.py.
    """
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    if not config_dir.is_dir():
        raise ValueError(
            f"Expected a directory path for config_dir, but got a file: {config_dir}"
        )

    config_path = config_dir / "config.yaml"
    utils_file = config_dir / "utils.py"
    config_file_name = (
        "config.yaml"  #  config file name is fixed now, for error message consistency
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_file_name} not found in {config_dir}")

    if not utils_file.exists():
        raise FileNotFoundError(
            f"utils.py not found in the same directory as {config_file_name}")

    # Read the config.yaml file
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"Error parsing YAML file: {config_file_name} in {config_dir} - {e}"
        ) from e

    func_ref = config.get("func")
    if not func_ref:
        raise ValueError(
            f"Key 'func' not found in {config_file_name} in {config_dir}")

    # Expecting a format like "utils.<function_name>"
    parts = func_ref.split(".")
    if len(parts) != 2 or parts[0] != "utils":
        raise ValueError(
            f"Value of 'func' in {config_file_name} in {config_dir} must be in the form"
            f"'utils.<function_name>', got '{func_ref}'")
    function_name = parts[1]

    # Dynamically import the utils module from utils.py
    spec = importlib.util.spec_from_file_location("utils", utils_file)
    if spec is None:
        raise ImportError(
            f"Could not create module spec for utils.py at {utils_file}")
    utils_module = importlib.util.module_from_spec(spec)
    if utils_module is None:
        raise ImportError(
            f"Could not create module from spec for utils.py at {utils_file}")
    try:
        spec.loader.exec_module(utils_module)
    except Exception as e:
        raise ImportError(
            f"Error executing utils.py at {utils_file}: {e}") from e

    func = getattr(
        utils_module, function_name
    )  # Remove None as default, will raise AttributeError if not found
    if not callable(func):
        raise ValueError(
            f"'{function_name}' from utils.py at {utils_file} is not callable."
        )

    return func


def create_task_path_dict(task_names: list[str], task_folders: list[Path],
                          logger) -> dict[str, Path]:
    """
    Creates a dictionary mapping task names to their corresponding folder paths
    based on the provided configuration files.
    This function iterates over given task folders, reading each folder's "config.yaml"
    to verify that the task name specified within the YAML
    matches the folder name. Two modes of operation are supported:
    - If "all" is present in task_names,
        the function processes every folder in task_folders.
    - Otherwise, only the folders with names that match the specified
        task_names are processed.
    Parameters:
        task_names (list[str]): A list of task names. If the string "all" is included,
            all available tasks in task_folders are processed.
        task_folders (list[Path]): A list of Path objects representing directories that
            potentially contain a "config.yaml" file.
    Returns:
        dict[str, Path]: A dictionary where keys are task names (str) and values are the
            corresponding folder Path objects.
    Raises:
        ValueError: If an error occurs while loading a YAML file or if the 'task_name'
            in the YAML file does not match the folder name.
    """

    task_path_dict = {}
    if "all" in task_names:
        logger.info("RUNNING ALL TASK", task_names=task_names)
        for folder in task_folders:
            yaml_path = folder / "config.yaml"
            if os.path.exists(yaml_path):
                with open(yaml_path) as stream:
                    try:
                        yaml_cfg = yaml.safe_load(stream)
                    except yaml.YAMLError as exc:
                        logger.error(exc)
                        raise ValueError(f"Error loading {yaml_path}") from exc
                if yaml_cfg["task_name"] == folder.name:
                    task_path_dict[folder.name] = folder
                else:
                    logger.error(
                        "task_name in yaml differs from folder name",
                        folder_name=folder.name,
                        yaml_task_name=yaml_cfg["task_name"],
                    )
                    raise ValueError(
                        f"The task_name in the yaml file {yaml_path} is "
                        f"different from the folder name {folder.name}")
    else:
        logger.info("RUNNING SPECIFIC TASKS", task_names=task_names)
        task_folder_names = [folder.name for folder in task_folders]
        for task_name in task_names:
            if task_name not in task_folder_names:
                logger.error(
                    "task_name not found in task_folders",
                    task_name=task_name,
                    task_folder_names=task_folder_names,
                )
                raise ValueError(
                    f"task '{task_name}' not found in folders: {task_folder_names}"
                )
            for folder in task_folders:
                if folder.name == task_name:
                    yaml_path = folder / "config.yaml"
                    if os.path.exists(yaml_path):
                        with open(yaml_path) as stream:
                            try:
                                yaml_cfg = yaml.safe_load(stream)
                            except yaml.YAMLError as exc:
                                logger.error(exc)
                                raise ValueError(
                                    f"Error loading {yaml_path}") from exc
                        if yaml_cfg["task_name"] == task_name:
                            task_path_dict[task_name] = folder
                        else:
                            logger.error(
                                "task_name in yaml differs from folder name",
                                folder_name=folder.name,
                                yaml_task_name=yaml_cfg["task_name"],
                            )
                            raise ValueError(
                                f"The task_name in the yaml file {yaml_path} is "
                                f"different from the folder name {folder.name}"
                            )
                    else:
                        logger.error(
                            "config.yaml not found in task folder",
                            task_name=task_name,
                            folder_path=folder,
                        )
                        raise ValueError(
                            f"config.yaml not found in task folder {folder}")
    return task_path_dict
