import json
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Optional, get_args

from babisteps.basemodels.complextracking import (ComplexTracking,
                                                  ObjectInLocationPolar,
                                                  ObjectInLocationWhat,
                                                  ObjectInLocationWhere,
                                                  ObjectsInLocation)
from babisteps.basemodels.nodes import Coordenate, Entity
from babisteps.basemodels.simpletracking import (ActorInLocationPolar,
                                                 ActorInLocationWhere,
                                                 ActorInLocationWho,
                                                 ActorWithObjectPolar,
                                                 ActorWithObjectWhat,
                                                 ActorWithObjectWho,
                                                 EntitiesInCoordenates,
                                                 SimpleTracker)
from babisteps.utils import simple_parse_args_string


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


def create_simpletracking(
    q_stories: int,
    states_qty: int,
    locations: list[str],
    actors: list[str],
    objects: list[str],
    question: str,
    answer: str,
    path: Path,
    verbosity,
    logger,
    gen_kwargs: Optional[str] = None,
):

    if gen_kwargs is not None:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        if gen_kwargs == "":
            gen_kwargs = None
        else:
            logger.info("Parsed generator kwargs", kwargs=gen_kwargs)

    # constraing to Actors in Location, then objects is empty
    if locations and actors and not objects:
        question_request = {
            "polar": ActorInLocationPolar,
            "who": ActorInLocationWho,
            "where": ActorInLocationWhere,
        }
        shape_str = ("locations", "actors")
        entities = [Entity(name=entity) for entity in actors]
        coordenates = [Coordenate(name=coordenate) for coordenate in locations]
        model = EntitiesInCoordenates(entities=entities,
                                      coordenates=coordenates)
    elif actors and objects and not locations:
        question_request = {
            "polar": ActorWithObjectPolar,
            "what": ActorWithObjectWhat,
            "who": ActorWithObjectWho,
        }
        shape_str = ("actors", "objects")
        entities = [Entity(name=entity) for entity in objects]
        coordenates = [Coordenate(name=coordenate) for coordenate in actors]
        model = EntitiesInCoordenates(entities=entities,
                                      coordenates=coordenates)

    request = question_request[question]
    folder_name = request.__name__
    assert question in question_request, f"Question {question} not valid"
    answer_options = list(get_args(request.__annotations__["answer"]))
    assert answer in answer_options, (
        f"Answer '{answer}' not valid for '{question}'",
        f"Valid answers are {answer_options}.",
    )

    folder_path = prepare_path(path, folder_name, logger)
    topic = request(answer=answer)

    jsonl_dataset = []
    txt_dataset = ""
    for s in range(q_stories):
        logger.info("Creating story", story=s)
        generator = SimpleTracker(
            model=deepcopy(model)._shuffle(),
            states_qty=states_qty,
            topic=topic,
            verbosity=verbosity,
            shape_str=shape_str,
            log_file=os.path.join(path, "logs.txt"),
            # add only if gen_kwargs is not None
            **gen_kwargs if gen_kwargs is not None else {},
        )
        generator.create_ontology()
        generator.create_fol()

        json = generator.story.create_json()
        json["id"] = s
        txt = generator.story.create_txt()
        jsonl_dataset.append(json)
        txt_dataset += txt
    logger.info("End of stories creation")
    return jsonl_dataset, txt_dataset, folder_path


def create_complextracking(
    q_stories: int,
    states_qty: int,
    locations: list[str],
    actors: list[str],
    objects: list[str],
    question: str,
    answer: str,
    path: Path,
    verbosity,
    logger,
    gen_kwargs: Optional[str] = None,
):

    if gen_kwargs is not None:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        if gen_kwargs == "":
            gen_kwargs = None
        else:
            logger.info("Parsed generator kwargs", kwargs=gen_kwargs)
    question_request = {
        "polar": ObjectInLocationPolar,
        "what": ObjectInLocationWhat,
        "where": ObjectInLocationWhere,
    }
    shape_str = ("locations", "actors", "objects")
    d0 = [Coordenate(name=coordenate) for coordenate in locations]
    d1 = [Coordenate(name=entity) for entity in actors]
    d2 = [Entity(name=entity) for entity in objects]
    model = ObjectsInLocation(dim0=d0, dim1=d1, dim2=d2)

    request = question_request[question]
    folder_name = request.__name__
    assert question in question_request, f"Question {question} not valid"
    answer_options = list(get_args(request.__annotations__["answer"]))
    assert answer in answer_options, (
        f"Answer '{answer}' not valid for '{question}'",
        f"Valid answers are {answer_options}.",
    )
    folder_path = prepare_path(path, folder_name, logger)
    topic = request(answer=answer)

    jsonl_dataset = []
    txt_dataset = ""
    max_retries = gen_kwargs.get("max_retries",
                                 10) if gen_kwargs is not None else 10
    for s in range(q_stories):
        logger.info("Creating story", story=s)
        retries = 0
        while retries < max_retries:
            try:
                generator = ComplexTracking(
                    model=deepcopy(model)._shuffle(),
                    states_qty=states_qty,
                    topic=topic,
                    verbosity=verbosity,
                    shape_str=shape_str,
                    log_file=os.path.join(path, "logs.txt"),
                    # add only if gen_kwargs is not None
                    **gen_kwargs if gen_kwargs is not None else {},
                )
                generator.create_ontology()
                generator.create_fol()
                break
            except Exception as e:
                logger.error("Error creating story", error=str(e))
                retries += 1
        if retries >= max_retries:
            logger.error("Max retries reached. Skipping story", story=s)
            raise Exception("Max retries reached. Skipping story")

        json = generator.story.create_json()
        txt = generator.story.create_txt()
        json["id"] = s
        jsonl_dataset.append(json)
        txt_dataset += txt
    logger.info("End of stories creation")
    return jsonl_dataset, txt_dataset, folder_path


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
        raise Exception("Max retries reached. Skipping story")

    json = g.get_json()
    txt = g.get_txt()
    return json, txt
