import random
from copy import deepcopy
from typing import Callable

from pydantic import BaseModel, ConfigDict
from sparse._sparse_array import SparseArray


class State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    am: SparseArray
    index: int


class Entity(BaseModel):
    name: str

    def __hash__(self):
        return hash(self.name)


class Coordenate(BaseModel):
    name: str

    def __hash__(self):
        return hash(self.name)


class UnitState(BaseModel):
    entity: Entity
    coordenate: Coordenate


class EntityInCoordenateState(State):
    am: SparseArray

    @property
    def attr_as_set(self):
        attr = []
        for unit in self.am:
            attr.append((unit.entity, unit.coordenate))
        return set(attr)

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
