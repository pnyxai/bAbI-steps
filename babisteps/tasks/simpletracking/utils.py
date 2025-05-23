import logging
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from babisteps.basemodels.generators import DELIM
from babisteps.basemodels.nodes import Coordenate, Entity
from babisteps.basemodels.simpletracking import (ActorInLocationPolar,
                                                 ActorInLocationWhere,
                                                 ActorInLocationWho,
                                                 ActorWithObjectPolar,
                                                 ActorWithObjectWhat,
                                                 ActorWithObjectWho,
                                                 EntitiesInCoordenates,
                                                 SimpleTracker)
from babisteps.proccesing import prepare_path
from babisteps.utils import generate_framework

yaml_path = Path(__file__).parent / "config.yaml"
task_leaf_list = [
    ActorInLocationPolar,
    ActorInLocationWho,
    ActorInLocationWhere,
    ActorWithObjectPolar,
    ActorWithObjectWhat,
    ActorWithObjectWho,
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
    n_entities = yaml_cfg.get("entities")
    n_coordenates = yaml_cfg.get("coordenates")

    def generator_func():
        for leaf, answer, count in framework:
            for i in range(count):
                gen_kwargs = yaml_cfg["gen_kwargs"]
                # Check if a case of Actorin Location, or Actor with Object
                if leaf in [
                        ActorInLocationPolar,
                        ActorInLocationWho,
                        ActorInLocationWhere,
                ]:
                    shape_str = ("locations", "actors")
                    entities_g = np.random.choice(total_actors,
                                                  size=n_entities,
                                                  replace=False).tolist()
                    coordenates_g = np.random.choice(total_locations,
                                                     size=n_coordenates,
                                                     replace=False).tolist()
                elif leaf in [
                        ActorWithObjectPolar,
                        ActorWithObjectWhat,
                        ActorWithObjectWho,
                ]:
                    shape_str = ("actors", "objects")
                    entities_g = np.random.choice(total_objects,
                                                  size=n_entities,
                                                  replace=False).tolist()
                    coordenates_g = np.random.choice(total_actors,
                                                     size=n_coordenates,
                                                     replace=False).tolist()

                entities = [Entity(name=entity) for entity in entities_g]
                coordenates = [
                    Coordenate(name=coordenate) for coordenate in coordenates_g
                ]
                model = EntitiesInCoordenates(entities=entities,
                                              coordenates=coordenates)
                runtime_name = leaf.__name__ + DELIM + answer + DELIM + str(i)
                topic = leaf(answer=answer)
                generator = SimpleTracker(
                    model=deepcopy(model)._shuffle(),
                    states_qty=states_qty,
                    topic=topic,
                    verbosity=verbosity,
                    shape_str=shape_str,
                    log_file=log_file,
                    name=runtime_name,
                    **gen_kwargs if gen_kwargs is not None else {},
                )
                yield generator

    return generator_func(), folder_path
