import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sparse._sparse_array import SparseArray

from babisteps.utils import logger


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


class Entity(BaseModel):
    name: str

    def __hash__(self):
        return hash(self.name)


class Coordenate(BaseModel):
    name: str

    def __hash__(self):
        return hash(self.name)


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
        entities = [int(key[1]) for key in entities.data]
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

    def create_transition(
        self,
        num_transitions: int,
        condition: Callable,
        axis: int,
        filter: Optional[Callable] = None,
    ) -> Union[SparseArray, None]:

        # TODO: This will be the class in charge of
        # make_actor_transition and make_object_transition
        
        pass
