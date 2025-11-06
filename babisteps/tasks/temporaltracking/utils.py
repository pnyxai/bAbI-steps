import logging
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from babisteps.basemodels.generators import DELIM
from babisteps.basemodels.nodes import Coordinate, Entity
from babisteps.basemodels.temporaltracking import (  # TemporalTrackingWhereRequest, ,
    EntitiesInCoordinatesAsEvents, TemporalTracking,
    TemporalTrackingPolarRequest, TemporalTrackingWhenRequest,
    TemporalTrackingWhereRequest, TemporalTrackingWhoRequest)
from babisteps.proccesing import prepare_path
from babisteps.tasks.immediateorder.utils import (_get_list_relations,
                                                  _get_relations_by_type)
from babisteps.utils import generate_framework

yaml_path = Path(__file__).parent / "config.yaml"
task_leaf_list = [
    TemporalTrackingPolarRequest,
    TemporalTrackingWhenRequest,
    TemporalTrackingWhereRequest,
    TemporalTrackingWhoRequest,
]


def _get_generators(**kwargs):
    # Commons
    num_samples = kwargs.get("num_samples_by_task")
    total_locations = kwargs.get("locations")
    total_actors = kwargs.get("actors")
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
    n_coordinates = yaml_cfg.get("coordinates")
    events_qty = yaml_cfg.get("events_qty")
    r_name_by_r_type = _get_relations_by_type(total_relations)

    # Re-read yaml_cfg and override `edge_qty` if specified there
    if "edge_qty" in yaml_cfg:
        edge_qty = yaml_cfg["edge_qty"]

    def generator_func(leaf, answer, i):
        gen_kwargs = yaml_cfg["gen_kwargs"]
        if leaf in [
                TemporalTrackingPolarRequest,
                TemporalTrackingWhenRequest,
                TemporalTrackingWhereRequest,
                TemporalTrackingWhoRequest,
        ]:
            # Temporal Order
            r_type_g = "relative_event"
            # Get relations compatible with the selected type
            relation = _get_list_relations(
                r_type_g,
                r_name_by_r_type,
                relations_qty,
                total_relations,
                relation_types_compatibility,
            )[0]
            shape_str = ("locations", "actors")
            entities_g = np.random.choice(total_actors,
                                          size=n_entities,
                                          replace=False).tolist()
            coordinates_g = np.random.choice(total_locations,
                                             size=n_coordinates,
                                             replace=False).tolist()
        # TODO: It should be extended for objects as entities, and actors as coordinates

        entities = [Entity(name=entity) for entity in entities_g]
        coordinates = [
            Coordinate(name=coordinate) for coordinate in coordinates_g
        ]
        model = EntitiesInCoordinatesAsEvents(entities=entities,
                                              coordinates=coordinates,
                                              relation=relation)
        runtime_name = leaf.__name__ + DELIM + answer + DELIM + str(i)
        topic = leaf(answer=answer,
                     r=relation,
                     relation_type=r_type_g,
                     shape_str=shape_str)
        generator = TemporalTracking(
            model=deepcopy(model)._shuffle(),
            events_qty=events_qty,
            edge_qty=edge_qty,
            topic=topic,
            verbosity=verbosity,
            log_file=log_file,
            name=runtime_name,
            **gen_kwargs if gen_kwargs is not None else {},
        )
        return generator

    return framework, generator_func, folder_path
