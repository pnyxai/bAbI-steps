import logging
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from babisteps.basemodels.generators import (DELIM, OrderModel,
                                             OrderRequestHow,
                                             OrderRequestPolar)
from babisteps.basemodels.nodes import Entity
from babisteps.basemodels.order import GeneralOrder
from babisteps.proccesing import prepare_path
from babisteps.tasks.immediateorder.utils import (
    _get_list_relations, _get_relations_by_type,
    relations_type_to_entities_dict)
from babisteps.utils import generate_framework

yaml_path = Path(__file__).parent / "config.yaml"
task_leaf_list = [
    OrderRequestPolar,
    OrderRequestHow,
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
    r_name_by_r_type = _get_relations_by_type(total_relations)

    def generator_func(leaf, answer, i):

        gen_kwargs = yaml_cfg["gen_kwargs"]
        r_type_g = "relative_size"
        # Get relations compatible with the selected type
        relations = _get_list_relations(
            r_type_g,
            r_name_by_r_type,
            relations_qty,
            total_relations,
            relation_types_compatibility,
        )
        # Get entities compatible with the selected relation type
        entity_type = random.choice(relations_type_to_entities_dict[r_type_g])
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
        generator = GeneralOrder(
            model=deepcopy(model)._shuffle(),
            edge_qty=edge_qty,
            topic=topic,
            verbosity=verbosity,
            log_file=log_file,
            name=runtime_name,
            **gen_kwargs if gen_kwargs is not None else {},
        )
        return generator

    return framework, generator_func, folder_path
