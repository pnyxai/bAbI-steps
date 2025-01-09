from pydantic import BaseModel
from typing import Any, List, Callable
import random
import copy

class State(BaseModel):
    entities: Any
    index: int


class ActorInLocationEntityState(BaseModel):
    actor: str
    location: str

class ActorInLocationState(State):
    entities: List[ActorInLocationEntityState]
    
    @property
    def entities_as_set(self):
        entities = []
        for entity in self.entities:
            entities.append((entity.actor, entity.location))
        return set(entities)
    
    def create_delta(self, num_transitions:int, locations:List[str], any_condition:Callable, all_condition:Callable) -> List[ActorInLocationEntityState]:
        """
        Creates a delta of actor-location pairs based on specified conditions.
        Args:
            num_transitions (int): The number of transitions (actor-location pairs) to create.
            locations (List[str]): A list of possible locations.
            any_condition (Callable): A callable that takes a pair (actor, location) and returns a boolean.
            all_condition (Callable): A callable that takes a pair (actor, location) and returns a boolean.
        Returns:
            List[ActorInLocationEntityState]: A list of ActorInLocationEntityState objects representing the delta.
        """

        oring_set = self.entities_as_set
        i = 0
        while True:
            delta = []
            # withou repetition
            entities = random.sample(list(oring_set), num_transitions) # get the actor list
            rnd_actors = [entity[0] for entity in entities]
            rnd_locations = random.choices([location for location in locations], k=num_transitions)
            aux_entities = {(actor, location) for actor, location in zip(rnd_actors, rnd_locations)}

            # create the delta
            for e in aux_entities:
                a, l = e[0], e[1]
                delta_i = ActorInLocationEntityState(actor=a, location=l)
                delta.append(delta_i)

            if any(any_condition(pair) for pair in aux_entities) and all(all_condition(pair) for pair in aux_entities):
                pass
            else:
                continue

            return delta

    def create_state_from_delta(self, j:int, delta:List[ActorInLocationEntityState]):

        new_entities = copy.deepcopy(self.entities)
        new_state = ActorInLocationState(entities = new_entities, index=j)
        for delta_i in delta:
            for entity in new_state.entities:
                if entity.actor == delta_i.actor:
                    entity.location = delta_i.location
        return new_state

    def get_actor_location(self, actor:str):
        for entity in self.entities:
            if entity.actor == actor:
                return entity.location
        return None


    






            