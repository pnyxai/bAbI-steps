import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import networkx as nx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sparse._sparse_array import SparseArray

import babisteps.operators as ops
from babisteps import logger


class State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    am: SparseArray
    index: int
    verbosity: Union[int, str] = Field(default=logging.INFO)
    logger: Optional[Any] = None
    log_file: Optional[Path] = None

    @model_validator(mode="after")
    def fill_logger(self):
        if not self.logger:
            self.logger = logger.get_logger(
                # use class name and index to create a unique logger name
                f"{self.__class__.__name__}_{self.index}",
                level=self.verbosity,
                log_file=self.log_file,
            )
        return self


class ImmediateGraph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    am: SparseArray
    g: nx.DiGraph
    name: str
    index: int

    @model_validator(mode="after")
    def _is_DAG(self):
        if not nx.is_directed_acyclic_graph(self.g):
            raise ValueError("The graph must be a Directed Acyclic Graph")
        return self


class Entity(BaseModel):
    name: str

    def __hash__(self):
        return hash(self.name)


class Coordenate(BaseModel):
    name: str

    def __hash__(self):
        return hash(self.name)


class Relationship(BaseModel):
    name: str
    r0: Optional[str] = None
    r1: Optional[str] = None

    def __hash__(self):
        return hash(self.name)

    @model_validator(mode="after")
    def _split_name(self):
        # validate that the name is in the form of "<r1>-<r2>"
        if "-" not in self.name:
            raise ValueError("The name must be in the form of '<r1>-<r2>'")
        self.r0, self.r1 = self.name.split("-")
        return self


class EntityInCoordenateState(State):
    am: SparseArray

    def create_transition(
        self,
        num_transitions: int,
        condition: Callable,
    ) -> SparseArray:
        """
        Creates a delta of actor-location pairs based on specified conditions.
        Args:
            num_transitions (int): The number of transitions (actor-location pairs) to
            create.
            coordenate (list[str]): A list of possible coordenate.
            condition (Callable): A callable that takes a pair (actor, location)
            and returns a boolean.
        Returns:
           SparseArray: A SparseArray objects representing the new state.
        """

        while True:
            next_am = deepcopy(self.am)
            # take k without repeat.
            e = random.sample(list(next_am.data.keys()), k=num_transitions)
            for i_e in e:
                x, y = i_e[0], i_e[1]
                # get a different coordenate different from current
                set_x = set([t for t in range(next_am.shape[0])]) - set([x])
                next_x = random.choice(list(set_x))
                next_am[next_x, y] = 1
                next_am[i_e] = 0
            # check if the delta satisfies the conditions
            if condition(next_am):
                pass
            else:
                continue

            f = self.validate_next(next_am)

            return next_am, f

    def get_entity_coordenate(self, entity: Entity):
        for unit in self.am:
            if unit.entity == entity:
                return unit.coordenate
        return None

    def get_entities_in_coodenate(self, coordenate: int):
        entities = []
        entities = self.am[coordenate, :] == 1
        entities = [int(key[0]) for key in entities.data]
        return entities

    def validate_next(self, next_am):
        # Due to reducen operations only work in COO sparse matrix
        array = next_am.to_coo().sum(axis=0).data
        flag = all(array == 1)
        return flag


class ObjectInLocationStatePolar(State):
    am: SparseArray

    def make_actor_transition(
        self,
        num_transitions: int,
        condition: Callable,
        filter: Optional[Callable] = None,
        limit=50,
    ) -> Union[SparseArray, None]:
        """
        Creates a delta of actor-location pairs based on specified conditions.
        Args:
            num_transitions (int):
            The number of transitions (actor-location pairs) to create.
            condition (Callable):
            A callable that takes a pair (actor, location) and returns a boolean.
        Returns:
           SparseArray: A SparseArray object representing the new state.
        """
        r = 0
        while r < limit:
            new_am = deepcopy(self.am)
            filtered_am = filter(new_am) if filter else new_am
            # avoid "nobodys" (:-1), such actor can move, and translate object.
            e = random.sample(list(filtered_am[:, :-1, :].data.keys()),
                              k=num_transitions)
            for i_e in e:
                x, y, _ = i_e  # Extract the current location (`x`) and actor (`y`)
                # Get all object indices (`z`) the actor (`y`) has at location (`x`)
                object_indices = [int(z_i[0]) for z_i in new_am[x, y].data]
                if not object_indices:
                    # Skip if the actor has no objects at the current location
                    continue
                # Pick a new location for the actor
                set_x = set(range(new_am.shape[0])) - {x}
                next_x = random.choice(list(set_x))
                # Move all objects the actor has from `x` to `next_x`
                for z in object_indices:
                    new_am[x, y, z] = 0  # Set to 0 at the current location
                    new_am[next_x, y, z] = 1  # Set to 1 at the new location
            if condition(new_am):
                pass
            else:
                r += 1
                continue
            f = self.validate_next(new_am)

            if not f:
                raise ValueError(
                    "validate_next do not aggregates axis (0,1) into [1,1,..,1]"
                )

            self.logger.debug(
                "Tx: actor",
                x=int(x),
                next_x=int(next_x),
                y=int(y),
                z=object_indices,
                retry=r,
            )
            return new_am

        return None

    def make_object_transition(
        self,
        num_transitions: int,
        condition: Callable,
        filter: Optional[Callable] = None,
        limit=50,
    ) -> Union[SparseArray, None]:
        """
        Creates a delta of object-actor pairs based on specified conditions.
        Args:
            num_transitions (int):The number of transitions (object-actor pairs)
            to create.
            condition (Callable): A callable that takes a pair (object, actor) and
            returns a boolean.
        Returns:
        SparseArray: A SparseArray object representing the new state.
        """
        t = 0
        while t < limit:
            new_am = deepcopy(self.am)
            filtered_am = filter(new_am) if filter else new_am
            # lets remove the nothing object, due to it can be given to any actor,
            # it just a placeholder
            filtered_am = filtered_am[:, :, :-1]
            # At this point, if this is empty, then none can give nothing to no nobody,
            # an actor transition should be done instead.
            if not filtered_am.data:
                self.logger.debug(
                    "Actors have nothing objects, moving into a actor transition"
                )
                return None
            # if there is only person, with nothing in this please, means that
            e = random.sample(list(filtered_am.data.keys()), k=num_transitions)
            valid_transition = True
            for i_e in e:
                x, y, z = i_e[0], i_e[1], i_e[2]
                # Ensure the new y is chosen from the same location
                set_y = new_am[x, :] == 1
                set_y = np.array(list(set_y.data.keys()), dtype=np.int64)
                # incluyde always the nobody actor such that item can be left there
                # but only if the actor has a unique object with him.
                set_y = np.unique(np.append(set_y.T[0], [new_am.shape[1] - 1]))
                set_y = set(set_y) - set([y])
                if not set_y:
                    e_converted = [
                        int(item) for sublist in e for item in sublist
                    ]
                    self.logger.debug("No valid transition",
                                      e=e_converted,
                                      set_y=set_y)
                    valid_transition = False
                    break  # Break out of the for loop and continue the while loop
                next_y = random.choice(list(set_y))
                new_am[x, next_y, z] = 1
                new_am[i_e] = 0
                # clean the nothing objet, to avoid errors
                new_am[x, next_y, -1] = 0
                if new_am[x, y, :-1].to_coo().sum() == 0:
                    self.logger.debug("Actor with 'nothing",
                                      x=int(x),
                                      y=int(y))
                    new_am[x, y, -1] = 1

            if not valid_transition:
                t += 1
                continue  # Continue the while loop if no valid actor is found
            if condition(new_am):
                pass
            else:
                t += 1
                continue

            self.logger.debug(
                "Tx: object",
                y=int(y),
                next_y=int(next_y),
                x=int(x),
                z=int(z),
                retry=t,
            )
            return new_am

        return None

    def create_transition(
        self,
        num_transitions: int,
        condition: Callable,
        axis: int,
        filter: Optional[Callable] = None,
    ) -> Union[SparseArray, None]:
        if axis == 1:
            new_am = self.make_actor_transition(num_transitions, condition,
                                                filter)
            if new_am is None:
                self.logger.debug(
                    "Fail make_actor_transition, now trying make_object_transition"
                )
                return self.make_object_transition(num_transitions, condition,
                                                   filter)
        elif axis == 2:
            new_am = self.make_object_transition(num_transitions, condition,
                                                 filter)
            if new_am is None:
                self.logger.debug(
                    "Fail make_object_transition, now trying make_actor_transition"
                )
                return self.make_actor_transition(num_transitions, condition,
                                                  filter)
        else:
            raise ValueError(
                f"Axis {axis} is not a valid axis to create a transition")
        return new_am

    def validate_next(self, new_am):
        # Due to reducen operations only work in COO sparse matrix.
        # only real object are checked, nothing (-1) can be haved by more than one actor
        array = np.sum(new_am[:, :, :-1].to_coo(), axis=(0, 1)).todense()
        flag = all(array == 1)
        return flag


class ObjectInLocationState(State):
    """This clase consider the What/Where question/answer pairs for ObjectInLocation BUT
    using anti-locations coordinates, instead of having only the 'nowhere' location.
    """

    am: SparseArray
    actor_locations_map: Optional[dict] = None
    objects_map: Optional[dict] = None
    p_nowhere_OR: Optional[float] = None
    method_p_nowhere_OR: Optional[Literal['fix', 'cap']] = None

    def get_actor_locations(self):
        """
        Processes a sparse matrix/adjacency matrix (state) 'x' with dimensions:
        The function:
        1. Slices the matrix to remove the 'nobody' actor (x[:, :-1, :])
        2. Converts the sliced matrix into a boolean matrix (==1)
        3. Iterates through the sparse representation and collects unique locations
            for each actor.
        4. Converts np.int64 values to Python int in the resulting dict.
        """
        # Slice out the 'nobody' actor and create a boolean matrix
        all_actors = self.am[:, :-1, :] == 1
        actor_locations = {}
        for (loc, actor, obj), flag in all_actors.data.items():
            if flag:  # only consider True entries
                # Convert np.int64 to int for actor and location
                actor_key = int(actor)
                location_val = int(loc)
                actor_locations.setdefault(actor_key, set()).add(location_val)
        # Convert each actor's set of locations to a sorted list (with python ints)
        map = {
            actor: sorted(list(locs))
            for actor, locs in actor_locations.items()
        }
        # sort the dictionary by key
        return dict(sorted(map.items()))

    def get_possibles_actor_transitions(self, location_to_locations_map):
        """
        This function return the list of possible transitions.
        Each element of the list is a dictionary with the actor index and the possible
        transition as value.
        """
        possible_actor_transition = []
        for a, loc in self.actor_locations_map.items():
            # check that loc is a key in the mapper, if not, skip
            if tuple(loc) not in location_to_locations_map:
                continue
            for possible_loc in location_to_locations_map[tuple(loc)]:
                possible_actor_transition.append({a: possible_loc})
        return possible_actor_transition

    def get_object_actor(self):
        """
        Processes a sparse matrix/adjacency matrix (state) 'x' with dimensions.
        Iterates through the sparse representation and collects actors' object.
        """
        all_items = self.am[:, :, :] == 1
        object_actor = {}
        for (loc, actor, obj), _ in all_items.data.items():
            # Convert np.int64 to int for actor and location
            actor_key = int(actor)
            obj_key = int(obj)
            object_actor.setdefault(actor_key, set()).add(obj_key)
        # Convert each actor's set of object to a sorted list (with python ints)
        map = {
            actor: sorted(list(objs))
            for actor, objs in object_actor.items()
        }
        # validate that if actors that has more that one object
        # there is not a 'nothing' object.
        nothing_object = self.am.shape[2] - 1
        for actor, objs in map.items():
            if len(objs) > 1 and nothing_object in objs:
                raise ValueError(
                    f"Actor {actor} has more than one object and has the 'nothing' object"  # noqa: E501
                )
        return dict(sorted(map.items()))

    def get_objects_map(self):
        """
        Return the actor-object and object-location dict given a sparse matrix.

        Args:
            x (sparse.DOK): Sparse matrix with the shape (locations, actors, objects)
        Returns:
        A dict with the following key:values
            actor_object (dict): Dictionary like {actor:list of objects}.
            object_location (dict): Dictionary like {object: location} owned by nobody.
        """
        # Slice out the 'nothing' object and create a boolean matrix
        all_items = self.am[:, :-1, :] == 1
        map = {}
        actor_object = {}
        for (loc, actor, obj), _ in all_items.data.items():
            actor_key = int(actor)
            obj_key = int(obj)
            actor_object.setdefault(actor_key, set()).add(obj_key)
        # Convert each actor's set of object to a sorted list (with python ints)
        actor_object = {
            actor: sorted(list(objs))
            for actor, objs in actor_object.items()
        }
        actor_object = dict(sorted(actor_object.items()))

        nobody_objects = self.am[:, -1, :] == 1
        object_location = {}
        for (loc, obj), _ in nobody_objects.data.items():
            location_key = int(loc)
            obj_key = int(obj)
            object_location.setdefault(obj_key, set()).add(location_key)

        map["actor_object"] = actor_object
        map["object_location"] = object_location
        return map

    def get_possible_objects_transitions(self):
        n_l = self.am.shape[0] // 2
        n_a = self.am.shape[1] - 1
        n_o = self.am.shape[2] - 1
        # iterate over each object
        # Create possible_transition_solution_list
        object_transitions = {}
        for z in range(n_o):
            filtered_am = self.am[:, :, z] == 1
            x_y = list(filtered_am.data.keys())[0]
            y = int(x_y[1])
            current_location = list(self.actor_locations_map[y] if y in
                                    self.actor_locations_map else self.
                                    objects_map["object_location"][z])
            set_y = [
                actor for actor, location in self.actor_locations_map.items()
                if location == current_location
            ]
            # if set_y is empty, then this mean that there is none actor in location of
            # the z object (who has the object is nobody!)
            if not set_y:
                continue
            # now, given the set_y, this means that there is at least one actor
            # (that at minimum is the one who has the object)
            # so, lets add the nobody actor to the set_y
            set_y.append(n_a)
            # and remove current owner of the object
            set_y.remove(y)
            v_position = np.zeros(n_l * 2, dtype=int)
            v_position[current_location] = 1
            parents_possibles = ops.generate_OR_parents(v_position)
            object_transitions[z] = {
                "owner": y,
                "current_location": current_location,
                "from_possible_parents": set_y,
                "possible_parents_locations": parents_possibles,
            }
        return object_transitions

    def _get_OR_parents(self, possible_parents_locations):
        """
        Funtion that retrieve one of the possible parents locations.
        IMPORTANT: It assumes that the NOWHERE location is the last one.
        When there is more that 1 possible parent location, it will take
        the NOWHERE with probability p.

        Args:
            possible_parents_locations (list): List of possible parents locations.

        Returns:
            owner_prev_loc (list): List of the previous owner location.
            parent_prev_loc (list): List of the previous parent location.
        
        """
        if len(possible_parents_locations) > 1:
            # TODO: Replace the p with a parameter that can be set by the user
            p = 1 / len(possible_parents_locations)
            if self.method_p_nowhere_OR == 'fix':
                p = self.p_nowhere_OR if self.p_nowhere_OR else p
            elif self.method_p_nowhere_OR == 'cap':
                p = min(p, self.p_nowhere_OR) if self.p_nowhere_OR else p

            #p = min(p, self.p_nowhere_OR) if self.p_nowhere_OR else p
            if random.random() < p:
                # 1. Pick the NOWHERE location
                possible_parents_location = possible_parents_locations[-1]
            else:
                # 2. Pick a random possible_parents_locations pairs
                # except the NOWHERE location
                possible_parents_location = random.choice(
                    possible_parents_locations[:-1])
        else:
            # 3. Pick the only possible parent location
            possible_parents_location = possible_parents_locations[0]

        # Assing x1,x2 OR to owner/previous_parent_rnd
        c = random.getrandbits(1)
        owner_prev_loc = np.array(possible_parents_location[c])
        owner_prev_loc = np.where(owner_prev_loc == 1)[0].tolist()
        parent_prev_loc = np.array(possible_parents_location[1 - c])
        parent_prev_loc = np.where(parent_prev_loc == 1)[0].tolist()
        return owner_prev_loc, parent_prev_loc

    def make_object_transition(
        self,
        obj_txs: dict,
        num_transitions: int,
        condition: Callable,
        location_to_locations_map,
        filter: Optional[Callable] = None,
        limit=50,
    ) -> tuple[Optional[State], Optional[list]]:
        """
        Creates a delta of object-actor pairs based on specified conditions.
        Args:
            num_transitions (int):The number of transitions (object-actor pairs)
            to create.
            condition (Callable): A callable that takes a pair (object, actor) and
            returns a boolean.
        Returns:
        SparseArray: A SparseArray object representing the new state.
        """
        new_am = deepcopy(self.am)
        obj_txs_tmp = deepcopy(obj_txs)
        t = 0
        nb = new_am.shape[1] - 1
        while t < limit:
            # 1. Pick a random key from obj_txs_tmp
            obj_rnd = random.choice(list(obj_txs_tmp.keys()))
            # 2. Pick a random actor to give that object to
            # from obj_txs_tmp[obj_rnd]["from_possible_parents"]
            previous_parent_rnd = random.choice(
                obj_txs_tmp[obj_rnd]["from_possible_parents"])
            current_loc = obj_txs_tmp[obj_rnd]["current_location"]
            owner = obj_txs_tmp[obj_rnd]["owner"]
            # 3. Pick a random possible_parents_locations pairs
            owner_prev_loc, parent_prev_loc = self._get_OR_parents(
                obj_txs_tmp[obj_rnd]["possible_parents_locations"])
            # 4. Remove obj_rnd from the current actor
            new_am[:, owner, obj_rnd] = 0
            # 5. Adjust objects position for both owner and previous_parent_rnd
            # this is done only for the object that is not the object to be moved
            # and for actors that are not `nobody``.
            for z in obj_txs:
                if z != obj_rnd:
                    if (obj_txs[z]["owner"] == previous_parent_rnd
                            and previous_parent_rnd != nb):
                        for x in current_loc:
                            new_am[x, previous_parent_rnd, z] = 0
                        for x in parent_prev_loc:
                            new_am[x, previous_parent_rnd, z] = 1
                    if obj_txs[z]["owner"] == owner and owner != nb:
                        for x in current_loc:
                            new_am[x, owner, z] = 0
                        for x in owner_prev_loc:
                            new_am[x, owner, z] = 1
            # 6. Place the object to the new actor
            for x in parent_prev_loc:
                new_am[x, previous_parent_rnd, obj_rnd] = 1

            # 7. Now, clean the `nothing` from previous_parent_rnd
            # (now it has an object!)
            new_am[:, previous_parent_rnd, -1] = 0
            # 8. If the owner has no objects, give them the 'nothing' object, except
            # if the owner is the nobody actor
            if new_am[:, owner, :-1].to_coo().sum() == 0 and owner != nb:
                # log that the owner has nothing
                self.logger.debug("Actor with 'nothing'",
                                  owner=owner,
                                  owner_prev_loc=owner_prev_loc)
                for x in owner_prev_loc:
                    new_am[x, owner, -1] = 1

            # 9 If the condition is met, return the new state
            if condition(new_am):
                new_state = ObjectInLocationState(
                    am=new_am,
                    p_nowhere_OR=self.p_nowhere_OR,
                    method_p_nowhere_OR=self.method_p_nowhere_OR,
                    index=self.index - 1,
                    verbosity=self.verbosity,
                    log_file=self.log_file,
                )
                # init necessary maps
                new_state.actor_locations_map = new_state.get_actor_locations()
                new_state.objects_map = new_state.get_objects_map()
                future_act_txs = new_state.get_possibles_actor_transitions(
                    location_to_locations_map)
                future_obj_txs = new_state.get_possible_objects_transitions()
                if len(future_act_txs) == 0 and len(future_obj_txs) == 0:
                    self.logger.warning("No possible transitions, retrying")
                    continue
                self.logger.debug(
                    "actor_locations_map",
                    actor_locations_map=new_state.actor_locations_map)
                self.logger.debug("new_state.objects_map",
                                  objects_map=new_state.objects_map)
                pass
            else:
                t += 1
                continue
            self.logger.info(
                "Tx: object",
                z=int(obj_rnd),
                loc=current_loc,
                owner=int(owner),
                owner_next_loc=owner_prev_loc,
                next_y=int(previous_parent_rnd),
                next_y_loc=parent_prev_loc,
                retry=t,
            )
            delta = [(2, 1), [int(previous_parent_rnd),
                              int(obj_rnd)], [int(owner),
                                              int(obj_rnd)]]
            return (new_state, delta)

        return (None, None)

    def make_actor_transition(
        self,
        act_txs: dict,
        num_transitions: int,
        condition: Callable,
        location_to_locations_map,
        filter: Optional[Callable] = None,
        limit=50,
    ) -> tuple[Optional[State], Optional[list]]:
        """
        Creates a delta of actor-location pairs based on specified conditions.
        Args:
            num_transitions (int):
            The number of transitions (actor-location pairs) to create.
            condition (Callable):
            A callable that takes a pair (actor, location) and returns a boolean.
        Returns:
           SparseArray: A SparseArray object representing the new state.
        """

        new_am = deepcopy(self.am)
        act_txs_tmp = deepcopy(act_txs)

        while act_txs_tmp:
            tx = random.choice(act_txs_tmp)
            for y, xs in tx.items():
                current_x = self.actor_locations_map[y]
                zs = self.objects_map["actor_object"][y]
                new_am[:, y, :] = 0
                for z in zs:
                    for x in xs:
                        new_am[x, y, z] = 1
            if condition(new_am):
                new_state = ObjectInLocationState(
                    am=new_am,
                    index=self.index - 1,
                    p_nowhere_OR=self.p_nowhere_OR,
                    method_p_nowhere_OR=self.method_p_nowhere_OR,
                    verbosity=self.verbosity,
                    log_file=self.log_file,
                )
                # init necessary maps
                new_state.actor_locations_map = new_state.get_actor_locations()
                new_state.objects_map = new_state.get_objects_map()
                future_act_txs = new_state.get_possibles_actor_transitions(
                    location_to_locations_map)
                future_obj_txs = new_state.get_possible_objects_transitions()
                if len(future_act_txs) == 0 and len(future_obj_txs) == 0:
                    self.logger.debug("No possible transitions, retrying")
                    continue
                self.logger.debug(
                    "actor_locations_map",
                    actor_locations_map=new_state.actor_locations_map)
                self.logger.debug("new_state.objects_map",
                                  objects_map=new_state.objects_map)
                pass
            else:
                # remove the tx from act_txs_tmp
                act_txs_tmp.remove(tx)
                continue
            # TODO: validate that the new_am is valid
            # f = self.validate_next(new_am)

            # if not f:
            #     raise ValueError(
            #         "validate_next do not aggregates axis (0,1) into [1,1,..,1]"
            #     )

            self.logger.info(
                "Tx: actor",
                y=y,
                x=current_x,
                next_x=xs,
                z=zs,
            )
            # Lets create the delta!
            delta = [(1, 0), [xs, int(y)], [current_x, int(y)]]
            return (new_state, delta)
        return (None, None)

    def create_transition(
        self,
        num_transitions: int,
        location_to_locations_map: dict,
        condition: Callable,
        axis: int,
        filter: Optional[Callable] = None,
    ) -> Union[State, None]:
        # init necessary maps
        new_state = None
        # Actor transition
        if axis == 1:
            act_txs = self.get_possibles_actor_transitions(
                location_to_locations_map)
            if act_txs:
                new_state, delta = self.make_actor_transition(
                    act_txs,
                    num_transitions,
                    condition,
                    location_to_locations_map,
                    filter,
                )
                if new_state is None:
                    self.logger.debug(
                        "Fail make_actor_transition, now trying make_object_transition"
                    )
                    obj_txs = self.get_possible_objects_transitions()
                    if obj_txs:
                        new_state, delta = self.make_object_transition(
                            obj_txs,
                            num_transitions,
                            condition,
                            location_to_locations_map,
                            filter,
                        )
                    else:
                        raise ValueError(
                            "Impossible to make any transition, FATAL!")
        # Object transition
        elif axis == 2:
            obj_txs = self.get_possible_objects_transitions()
            if obj_txs:
                new_state, delta = self.make_object_transition(
                    obj_txs,
                    num_transitions,
                    condition,
                    location_to_locations_map,
                    filter,
                )
                if new_state is None:
                    self.logger.debug(
                        "Fail make_object_transition, now trying make_actor_transition"
                    )
                    act_txs = self.get_possibles_actor_transitions(
                        location_to_locations_map)
                    if act_txs:
                        new_state, delta = self.make_actor_transition(
                            act_txs,
                            num_transitions,
                            condition,
                            location_to_locations_map,
                            filter,
                        )
                    else:
                        raise ValueError(
                            "Impossible to make any transition, FATAL!")
        else:
            raise ValueError(
                f"Axis {axis} is not a valid axis to create a transition")

        if new_state is None:
            self.logger.error(
                "Impossible to make any kind of transition, FATAL!", axis=axis)
            raise ValueError(
                "Impossible to make any kind of transition, FATAL!")
        return new_state, delta
