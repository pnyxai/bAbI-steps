import copy
import random
from typing import Any, Callable

from pydantic import BaseModel


class State(BaseModel):
    attr: Any
    index: int


class Entity(BaseModel):
    name: str
    type: str

    def __hash__(self):
        return hash((self.name, self.type))


class Coordenate(BaseModel):
    name: str
    type: str

    def __hash__(self):
        return hash((self.name, self.type))


class UnitState(BaseModel):
    entity: Entity
    coordenate: Coordenate


class EntityInLocationState(State):
    attr: list[UnitState]

    @property
    def attr_as_set(self):
        attr = []
        for unit in self.attr:
            attr.append((unit.entity, unit.coordenate))
        return set(attr)

    def create_delta(
        self,
        num_transitions: int,
        coordenate: list[str],
        any_condition: Callable,
        all_condition: Callable,
    ) -> list[UnitState]:
        """
        Creates a delta of actor-location pairs based on specified conditions.
        Args:
            num_transitions (int): The number of transitions (actor-location pairs) to 
            create.
            coordenate (list[str]): A list of possible coordenate.
            any_condition (Callable): A callable that takes a pair (actor, location) 
            and returns a boolean.
            all_condition (Callable): A callable that takes a pair (actor, location) 
            and returns a boolean.
        Returns:
            list[UnitState]: A list of UnitState objects representing the delta.
        """

        original_set = self.attr_as_set
        while True:
            delta = []
            # withou repetition
            attr = random.sample(list(original_set),
                                 num_transitions)  # get the actor list
            rnd_entities = [unit[0] for unit in attr]  # [0] get the entity
            rnd_coordenates = random.choices([coord for coord in coordenate],
                                             k=num_transitions)
            aux_attr = {(entity, coord)
                        for entity, coord in zip(rnd_entities, rnd_coordenates)
                        }
            # create deltas
            for u in aux_attr:
                e, c = u[0], u[1]
                delta_i = UnitState(entity=e, coordenate=c)
                delta.append(delta_i)
            # check if the delta satisfies the conditions
            if any(any_condition(pair) for pair in aux_attr) and all(
                    all_condition(pair) for pair in aux_attr):
                pass
            else:
                continue

            return delta

    def create_state_from_delta(self, j: int, delta: list[UnitState]):
        new_attr = copy.deepcopy(self.attr)
        new_state = EntityInLocationState(attr=new_attr, index=j)
        for delta_i in delta:
            for unit in new_state.attr:
                if unit.entity == delta_i.entity:
                    unit.coordenate = delta_i.coordenate
        return new_state

    def get_entity_coordenate(self, entity: Entity):
        for unit in self.attr:
            if unit.entity == entity:
                return unit.coordenate
        return None
