
import logging
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from babisteps import logger

from babisteps.basemodels.generators import (
    DELIM,
    OrderModel,
)
from babisteps.basemodels.pathfinding import PathFindingRequestWhich, PathFinding
from babisteps.basemodels.pathfinding import PathFinding
from babisteps.basemodels.nodes import Entity
from babisteps.proccesing import prepare_path
from babisteps.utils import generate_framework
from babisteps.tasks.immediateorder.utils import (
    _get_relations_by_type, _get_list_relations
)

yaml_path = Path(__file__).parent / "config.yaml"
task_leaf_list = [
    PathFindingRequestWhich,
]

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
    max_path_length = yaml_cfg.get("max_path_length")
    if max_path_length < 3:
        raise ValueError(f"max_path_length ({max_path_length}) must be at least 3")
    r_name_by_r_type = _get_relations_by_type(total_relations)
    # NOTE: PathFinding only with absolute_position
    # remove all keys from r_name_by_r_type except "absolute_position"
    r_name_by_r_type = {k: v for k, v in r_name_by_r_type.items() if k == "absolute_position"}    


    def generator_func():
        for leaf, answer, count in framework:
            for i in range(count):
                path_length = np.random.randint(3, max_path_length + 1)
                if not isinstance(answer, str):
                    answer = path_length
                gen_kwargs = yaml_cfg["gen_kwargs"]
                # PATHFINDING: currently only with absolute_position
                r_type_g = "absolute_position"
                # Get relations compatible with the selected type
                relations = _get_list_relations(
                    r_type_g,
                    r_name_by_r_type,
                    relations_qty,
                    total_relations,
                    relation_types_compatibility,
                )
                # Get entities compatible with the selected relation type
                entity_type = "locations" # NOTE: Hardcoded for PathFinding
                local_entities = kwargs.get(entity_type)
                entities = np.random.choice(local_entities,
                                            size=n_entities,
                                            replace=False).tolist()
                entities = [Entity(name=entity) for entity in entities]

                # Create the model
                model = OrderModel(entities=entities,
                                            relations=relations)
                runtime_name = leaf.__name__ + DELIM + str(
                    answer) + DELIM + str(i)
                # Complete the topic
                topic = leaf(
                    answer=answer,
                    relation_type=relations[0].relation_type,
                    shape_str=(entity_type, ),
                )
                # Create the generator
                generator = PathFinding(
                    model=deepcopy(model)._shuffle(),
                    edge_qty=edge_qty,
                    topic=topic,
                    verbosity=verbosity,
                    log_file=log_file,
                    name=runtime_name,
                    path_length=path_length,
                    **gen_kwargs if gen_kwargs is not None else {},
                )
                yield generator

    return generator_func(), folder_path
