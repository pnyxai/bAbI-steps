import logging
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from babisteps.basemodels.complextracking import (ComplexTracking,
                                                  ObjectInLocationPolar,
                                                  ObjectInLocationWhat,
                                                  ObjectInLocationWhere,
                                                  ObjectsInLocation)
from babisteps.basemodels.generators import DELIM
from babisteps.basemodels.nodes import Coordenate, Entity
from babisteps.proccesing import prepare_path
from babisteps.utils import generate_framework

yaml_path = Path(__file__).parent / "config.yaml"
task_leaf_list = [
    ObjectInLocationPolar, ObjectInLocationWhat, ObjectInLocationWhere
]


def _get_generators(**kwargs):
    # Commons
    num_samples = kwargs.get("num_samples_by_task")
    total_locations = kwargs.get("locations")
    total_actors = kwargs.get("actors")
    total_objects = kwargs.get("objects")
    output_path = kwargs.get("output_path")
    states_qty = kwargs.get("states_qty")
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
    shape_str = ("locations", "actors", "objects")
    n_locations = yaml_cfg.get("locations")
    n_actors = yaml_cfg.get("actors")
    n_objects = yaml_cfg.get("objects")

    def generator_func(leaf, answer, i):
        gen_kwargs = yaml_cfg["gen_kwargs"]
        locations = np.random.choice(total_locations,
                                     size=n_locations,
                                     replace=False).tolist()
        actors = np.random.choice(total_actors, size=n_actors,
                                  replace=False).tolist()
        objects = np.random.choice(total_objects,
                                   size=n_objects,
                                   replace=False).tolist()

        d0 = [Coordenate(name=coordenate) for coordenate in locations]
        d1 = [Coordenate(name=entity) for entity in actors]
        d2 = [Entity(name=entity) for entity in objects]
        model = ObjectsInLocation(dim0=d0, dim1=d1, dim2=d2)
        runtime_name = leaf.__name__ + DELIM + answer + DELIM + str(i)
        topic = leaf(answer=answer)
        generator = ComplexTracking(
            model=deepcopy(model)._shuffle(),
            states_qty=states_qty,
            topic=topic,
            verbosity=verbosity,
            shape_str=shape_str,
            log_file=log_file,
            name=runtime_name,
            **gen_kwargs if gen_kwargs is not None else {},
        )
        return generator

    return framework, generator_func, folder_path
