import logging
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import get_type_hints

import numpy as np
import yaml

from babisteps.basemodels.generators import (DELIM, OrderModel,
                                             OrderRequestHow,
                                             OrderRequestPolar,
                                             OrderRequestWhat)
from babisteps.basemodels.immediateorder import ImmediateOrder
from babisteps.basemodels.nodes import Entity, Relationship
from babisteps.proccesing import prepare_path
from babisteps.utils import generate_framework

yaml_path = Path(__file__).parent / "config.yaml"
task_leaf_list = [
    OrderRequestPolar,
    OrderRequestHow,
    OrderRequestWhat,
]

relations_type_to_entities_dict = {
    "relative_event": ["events"],
    "relative_size": ["locations", "actors", "objects"],
    "absolute_position": ["locations", "actors", "objects"],
    "relative_position": ["locations", "actors", "objects"],
}


def _get_relations_by_type(relations: dict):
    """
    Return a dictionary mapping each relation type to a list of relation.
    Parameters:
        relations (dict): A dictionary where each key is a relation identifier and each
         value is a dictionary containing at least a "type" key that indicates the 
         type of the relation.
    Returns:
        relations_by_type: A dictionary where each key is a unique relation type found
        in relations and each
              value is a list of relation identifiers that have that type.
    """

    relations_by_type = {}
    for relation in relations:
        if relations[relation]["type"] not in relations_by_type:
            relations_by_type[relations[relation]["type"]] = []
        relations_by_type[relations[relation]["type"]].append(relation)
    return relations_by_type


def _get_list_relations(
    relation_type,
    relation_names_by_type,
    relations_qty,
    total_relations,
    relation_types_compatibility,
):
    """
    Creates a list of compatible relations:

    1. Select a primary relation from the given relation type
    2. Find compatible relation types and gather their relations
    3. Remove the primary relation from the compatible list
    4. Select additional relations to reach the requested quantity

    Args:
        relation_type: The primary relation type to select from
        relation_names_by_type: Dictionary mapping relation types to relation names
        relations_qty: Number of relations to return
        total_relations: Dictionary of all relation definitions
        relation_types_compatibility: Dictionary of compatible relation types

    Returns:
        List of Relationship objects
    """

    # Create a helper function to avoid code duplication
    def create_relationship(rel_name):
        r = Relationship(
            name=rel_name,
            base=total_relations[rel_name]["base"],
            opposite=total_relations[rel_name]["opposite"],
            relation_type=total_relations[rel_name]["type"],
        )
        return r

    # Select primary relation
    primary_relation = random.choice(relation_names_by_type[relation_type])
    relations = [create_relationship(primary_relation)]

    # Get compatible relation types
    compatible_types = deepcopy(relation_types_compatibility[relation_type])

    # Handle mutually exclusive relation types
    if ("relative_position" in compatible_types
            and "absolute_position" in compatible_types):
        to_remove = random.choice(["relative_position", "absolute_position"])
        compatible_types.remove(to_remove)

    # Gather all compatible relations
    compatible_relations = []
    for comp_type in compatible_types:
        if comp_type in relation_names_by_type:
            compatible_relations.extend(relation_names_by_type[comp_type])

    # Remove the primary relation
    compatible_relations.remove(primary_relation)

    # Ensure we have enough compatible relations
    if len(compatible_relations) < relations_qty - 1:
        available = len(compatible_relations) + 1
        raise ValueError(
            f"Not enough compatible relations. Requested {relations_qty} "
            f"but only {available} available.")

    # Select additional relations
    for _ in range(relations_qty - 1):
        if not compatible_relations:
            break

        selected = random.choice(compatible_relations)
        relations.append(create_relationship(selected))
        compatible_relations.remove(selected)

    return relations


def _get_generators(**kwargs):
    # Commons
    num_samples = kwargs.get("num_samples_by_task")
    total_relations = kwargs.get("relations")
    relation_types_compatibility = kwargs.get("relation_types_compatibility")
    output_path = kwargs.get("output_path")
    edge_qty = kwargs.get("edges_qty")
    verbosity = getattr(logging, f"{kwargs.get('verbosity')}")
    # Task
    with open(yaml_path) as stream:
        try:
            yaml_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    framework = generate_framework(num_samples, task_leaf_list)
    folder_name = yaml_cfg["task_name"]
    folder_path = prepare_path(output_path, folder_name)
    log_file = os.path.join(folder_path, "logs.txt")
    relations_qty = yaml_cfg.get("relations_qty")
    n_entities = yaml_cfg.get("entities")
    r_name_by_r_type = _get_relations_by_type(total_relations)

    def generator_func():
        for leaf, answer, count in framework:
            for i in range(count):
                gen_kwargs = yaml_cfg["gen_kwargs"]
                # Pick a random relation type
                r_type_g = random.choice(
                    list(get_type_hints(leaf)["relation_type"].__args__))
                # Get relations compatible with the selected type
                relations = _get_list_relations(
                    r_type_g,
                    r_name_by_r_type,
                    relations_qty,
                    total_relations,
                    relation_types_compatibility,
                )
                # Get entities compatible with the selected relation type
                entity_type = random.choice(
                    relations_type_to_entities_dict[r_type_g])
                local_entities = kwargs.get(entity_type)
                entities = np.random.choice(local_entities,
                                            size=n_entities,
                                            replace=False).tolist()
                entities = [Entity(name=entity) for entity in entities]

                # Create the model
                model = OrderModel(entities=entities, relations=relations)
                runtime_name = leaf.__name__ + DELIM + answer + DELIM + str(i)
                # Complete the topic
                topic = leaf(
                    answer=answer,
                    relation_type=relations[0].relation_type,
                    shape_str=(entity_type, ),
                )
                # Create the generator
                generator = ImmediateOrder(
                    model=deepcopy(model)._shuffle(),
                    edge_qty=edge_qty,
                    topic=topic,
                    verbosity=verbosity,
                    log_file=log_file,
                    name=runtime_name,
                    **gen_kwargs if gen_kwargs is not None else {},
                )
                yield generator

    return generator_func(), folder_path
