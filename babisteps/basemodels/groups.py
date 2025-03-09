import numpy as np
from typing import Optional, Union
from pydantic import BaseModel
from copy import deepcopy
from babisteps.basemodels.nodes import State


class Group(BaseModel):
    loc: list
    actors: Optional[list] = []
    objects: Optional[list] = []


def _check_if_isin_group(
    groups: list[Group], actor: int, object: int, nobody_index: int
) -> Union[bool, int]:
    flag = False
    if actor == nobody_index:
        for i, g in enumerate(groups):
            if object in g.objects:
                flag = i
                break
    else:
        for i, g in enumerate(groups):
            if actor in g.actors:
                flag = i
                break
    return flag


def _create_group(
    actor_locations_map,
    objects_map,
    actor,
    object,
    nobody_index,
    source: bool,
    loc_extra_info,
) -> Union[Group, None]:
    if actor == nobody_index:
        if source:
            loc = objects_map["object_location"][object]
            actor_group = Group(
                loc=loc,
                objects=[object],
                actors=[],
            )
        else:
            loc = loc_extra_info
            actor_group = Group(
                loc=loc,
                objects=[object],
                actors=[],
            )
    else:
        loc = actor_locations_map[actor]
        actor_group = Group(
            loc=loc,
            actors=[actor],
        )
    return actor_group


def _retrieve_group(
    groups: list[Group],
    actor_locations_map: dict,
    objects_map: dict,
    actor: int,
    object: int,
    nobody_index: int,
    source: bool,
    loc_extra_info=None,
) -> Union[Group, int]:
    actor_in_group = _check_if_isin_group(groups, actor, object, nobody_index)
    if actor_in_group is False:
        actor_group = _create_group(
            actor_locations_map,
            objects_map,
            actor,
            object,
            nobody_index,
            source,
            loc_extra_info,
        )
        return actor_group
    else:
        return actor_in_group


def _get_object_transaction_OR_result(n_locs: int, y_i_group: Group, y_f_group: Group):
    """
    Given a state s_i and actros y_i and y_f, it returns the location vector results of
    the interaction.

    Args:
    n_locs: number of total locations (known + anti-locs)
    y_i_group: Group object
    y_f_group: Group object

    Returns:
    np.ndarray: location vector
    """
    known_locs = n_locs // 2
    y_i_bin = np.zeros(n_locs, dtype=bool)
    y_f_bin = np.zeros(n_locs, dtype=bool)
    y_i_bin[y_i_group.loc] = 1
    y_f_bin[y_f_group.loc] = 1
    # check if the second half of each group all 1s using bool conversion
    y_i_nowhere = bool(np.all(y_i_bin[known_locs:]))
    y_f_nowhere = bool(np.all(y_f_bin[known_locs:]))
    # Both NOWHERE -> return NOWHERE
    if y_i_nowhere and y_f_nowhere:
        # return the index where the y_i_bin is 1
        return np.where(y_i_bin == 1)[0]
    # One of them is NOWHERE -> return the other
    if y_i_nowhere:
        return np.where(y_f_bin == 1)[0]
    if y_f_nowhere:
        return np.where(y_i_bin == 1)[0]
    # # Any is NOWHERE -> peform AND
    # if y_i_nowhere or y_f_nowhere:
    #     _and = y_i_bin & y_f_bin
    #     return np.where(_and == 1)[0]

    # Both NOT NOWHERE -> do OR
    _or = y_i_bin | y_f_bin
    # Check result
    first_half = _or[:known_locs]
    second_half = _or[known_locs:]
    if not any(first_half):
        if np.sum(second_half) == known_locs - 1:
            # When the or result in a known location
            idx = np.where(second_half == 0)[0]
            _or[idx] = 1
    else:
        # The result is the OR betwen some anti-locations and some known location.
        assert np.sum(first_half) == 1, "There should be only one 1 in the first half"
        assert np.sum(second_half) == known_locs - 1, (
            "There should be only one 0 in the second half"
        )
    assert not bool(np.all(_or[known_locs:])), (
        "there can't be all 1s in the second half"
    )
    return np.where(_or == 1)[0]


def _get_forward(
    states: list[State],
    deltas: list,
    n_locs: int,
    nobody: int,
    logger,
):
    groups = []
    actor_locations_map = deepcopy(states[0].actor_locations_map)
    objects_map = deepcopy(states[0].objects_map)
    logger.debug(
        "Forward checking (initialization)",
        actor_locations_map=actor_locations_map,
        objects_map=objects_map,
    )
    try:
        for i, _ in enumerate(states[1:]):
            d = deltas[i]
            tx = d[0]
            if tx == (1, 0):
                # x_i = d[1][0]
                x_f = d[2][0]
                y_i = d[1][1]
                # check if the actor is in a group
                actor_in_group = _check_if_isin_group(groups, y_i, -1, -1)
                # if actor actor_in_group not False
                if actor_in_group is not False:
                    # remove the object from the group
                    groups[actor_in_group].actors.remove(y_i)
                    # if the group is empty, remove it
                    if (
                        not groups[actor_in_group].actors
                        and not groups[actor_in_group].objects
                    ):
                        groups.pop(actor_in_group)

                # time to update the actor location
                actor_locations_map[y_i] = x_f

            elif tx == (2, 1):
                z = d[1][1]
                y_i = d[1][0]
                y_f = d[2][0]

                ############################
                # GROUPS RETRIEVAL
                ############################
                if y_i == nobody:
                    y_i_group_index = _retrieve_group(
                        groups,
                        actor_locations_map,
                        objects_map,
                        y_i,
                        z,
                        nobody,
                        source=True,
                    )
                else:
                    y_i_group_index = _retrieve_group(
                        groups,
                        actor_locations_map,
                        objects_map,
                        y_i,
                        z,
                        nobody,
                        source=True,
                    )
                if isinstance(y_i_group_index, Group):
                    y_i_group = y_i_group_index
                elif isinstance(y_i_group_index, int):
                    y_i_group = groups.pop(y_i_group_index)
                else:
                    raise ValueError(f"Invalid type {type(y_i_group_index)}")

                if y_f == nobody:
                    y_f_group_index = _retrieve_group(
                        groups,
                        actor_locations_map,
                        objects_map,
                        y_f,
                        z,
                        nobody,
                        source=False,
                        loc_extra_info=y_i_group.loc,
                    )
                else:
                    y_f_group_index = _retrieve_group(
                        groups,
                        actor_locations_map,
                        objects_map,
                        y_f,
                        z,
                        nobody,
                        source=False,
                    )
                if isinstance(y_f_group_index, Group):
                    y_f_group = y_f_group_index
                elif isinstance(y_f_group_index, int):
                    y_f_group = groups.pop(y_f_group_index)
                else:
                    raise ValueError(f"Invalid type {type(y_f_group_index)}")

                ############################
                # GROUPS NEW LOCATION
                ############################
                new_group_loc = _get_object_transaction_OR_result(
                    n_locs, y_i_group, y_f_group
                )
                new_group_loc = new_group_loc.tolist()
                ############################
                # GROUP UNION
                ############################
                if y_i != nobody:
                    # If y_i is not nobody, then the child group is the union of y_i_group and y_f_group
                    # Note: this includes the case where y_i is nobody!
                    new_actors = list(set(y_i_group.actors + y_f_group.actors))
                    new_objects = list(set(y_i_group.objects + y_f_group.objects))
                    child_group = Group(
                        loc=new_group_loc,
                        actors=new_actors,
                        objects=new_objects,
                    )
                # When y_i is nobody, first remove from current_y_i_group_loc the object z, then add
                # actors and objects from y_f_group
                elif y_i == nobody:
                    # uniques actors and objects
                    new_actors = list(set(y_i_group.actors + y_f_group.actors))
                    new_objects = list(set(y_i_group.objects + y_f_group.objects))
                    # remove z from objects
                    new_objects.remove(z)
                    child_group = Group(
                        loc=new_group_loc,
                        actors=new_actors,
                        objects=new_objects,
                    )

                groups.append(child_group)
                # Update actor locations map
                for actor in child_group.actors:
                    actor_locations_map[actor] = child_group.loc
                # Update object locations map
                for obj in child_group.objects:
                    if obj in objects_map["object_location"]:
                        objects_map["object_location"][obj] = child_group.loc

                if y_i != nobody and y_f != nobody:
                    if y_f in objects_map["actor_object"]:
                        objects_map["actor_object"][y_f].append(z)
                        objects_map["actor_object"][y_f].sort()
                    else:
                        objects_map["actor_object"][y_f] = [z]

                    objects_map["actor_object"][y_i].remove(z)

                    if not objects_map["actor_object"][y_i]:
                        objects_map["actor_object"].pop(y_i)

                elif y_i == nobody:
                    objects_map["object_location"].pop(z)

                    if y_f in objects_map["actor_object"]:
                        objects_map["actor_object"][y_f].append(z)
                        objects_map["actor_object"][y_f].sort()
                    else:
                        objects_map["actor_object"][y_f] = [z]

                elif y_f == nobody:
                    objects_map["object_location"][z] = set(child_group.loc)
                    objects_map["actor_object"][y_i].remove(z)
                    if not objects_map["actor_object"][y_i]:
                        objects_map["actor_object"].pop(y_i)

            else:
                raise ValueError(f"Invalid Tx type {d[0]}")

            logger.debug(
                "Forward checking",
                i=i,
                delta=d,
                actor_locations_map=actor_locations_map,
                objects_map=objects_map,
                groups=groups,
            )
        return groups, actor_locations_map, objects_map, None, None
    except Exception as e:
        return groups, actor_locations_map, objects_map, i, e