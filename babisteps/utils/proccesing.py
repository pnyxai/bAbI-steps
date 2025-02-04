import json
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import get_args

from babisteps.basemodels.generators import (ActorInLocationPolar,
                                             ActorInLocationWhere,
                                             ActorInLocationWho,
                                             ActorWithObjectPolar,
                                             ActorWithObjectWhat,
                                             ActorWithObjectWho,
                                             ComplexTracking,
                                             EntitiesInCoordenates,
                                             ObjectInLocationPolar,
                                             ObjectsInLocation, SimpleTracker)
from babisteps.basemodels.nodes import Coordenate, Entity


def prepare_path(path: Path, folder_name: str, logger):
    # Define the folder path
    folder_path = path / folder_name  # Assuming `path` is a Path object
    # Check if the folder exists
    if folder_path.exists():
        logger.info("Clearing content", folder=folder_path)

        # Remove all contents inside the folder
        for item in folder_path.iterdir():
            if item.is_file() or item.is_symlink():
                logger.info("Deleting file", file=item)
                item.unlink()  # Remove file or symlink
            elif item.is_dir():
                logger.info("Deleting folder and its contents", folder=item)
                shutil.rmtree(item)  # Remove directory and its contents
    else:
        logger.info("Folder does not exist. Creating it.", folder=folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
    logger.info("Folder is now ready for use.", folder=folder_path)
    return folder_path


def save_as_jsonl(json_list,
                  folder_path: Path,
                  logger,
                  filename="output.jsonl"):
    """
    Saves a list of JSON objects as a JSONL file inside the specified folder.

    :param json_list: List of dictionaries to save
    :param folder_path: Path object representing the target folder
    :param filename: Name of the JSONL file (default: output.jsonl)
    """
    try:
        file_path = folder_path / filename  # Construct full file path

        with open(file_path, "w", encoding="utf-8") as f:
            for obj in json_list:
                f.write(json.dumps(obj) +
                        "\n")  # Write each JSON object on a new line
        logger.info("Saved JSONL", file_path=file_path)  # Print confirmation
    except Exception as e:
        logger.error("Error saving JSONL", error=str(e))
        raise e


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
        logger.info("Saved .txt", file_path=file_path)  # Print confirmation
    except Exception as e:
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
):
    # constraing to Actors in Location, then objects is empty
    if locations and actors and not objects:
        question_request = {
            "polar": ActorInLocationPolar,
            "who": ActorInLocationWho,
            "where": ActorInLocationWhere,
        }
        shape_str = ("Location", "Actor")
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
        shape_str = ("Actor", "Object")
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
):
    question_request = {
        "polar": ObjectInLocationPolar,
    }
    shape_str = ("Location", "Actor", "Object")
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
    for s in range(q_stories):
        logger.info("Creating story", story=s)
        generator = ComplexTracking(
            model=deepcopy(model)._shuffle(),
            states_qty=states_qty,
            topic=topic,
            verbosity=verbosity,
            shape_str=shape_str,
            log_file=os.path.join(path, "logs.txt"),
        )
        generator.create_ontology()
        generator.create_fol()

        json = generator.story.create_json()
        txt = generator.story.create_txt()
        json["id"] = s
        jsonl_dataset.append(json)
        txt_dataset += txt
    logger.info("End of stories creation")
    return jsonl_dataset, txt_dataset, folder_path
