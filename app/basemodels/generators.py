from pydantic import BaseModel, Field, model_validator
from typing import Optional, Any, List, Union, Callable

from typing_extensions import Annotated
from basemodels.nodes import State, ActorInLocationEntityState, ActorInLocationState
import random
import logging
from utils import logger

from locations import NW

class Generator(BaseModel):
    entities: Any
    states_qty: int
    states: Optional[List[State]] = None
    deltas: Optional[Any] = None

    def create_ontology(self):
        pass

    def create_fol(self):
        pass
    
    def create_nlp(self):
        pass

    def create_deltas(self):
        pass


class ActorsInLocationsEntities(BaseModel):
    actors: List[str]
    locations: List[str]

class ActorsInLocaltionsPolar(Generator):
    entities: ActorsInLocationsEntities
    answer: Annotated[int, Field(strict=True, ge=0, le=2)]
    verbosity: Union[int, str] = Field(default=logging.INFO)
    logger: Optional[Any] = None
    min_transitions: int = Field(ge=1)
    max_transitions: int = Field(ge=1)
    states: Optional[List[ActorInLocationEntityState]] = None
    fol: Optional[Any] = None
    uncertainty: str = 'NW'

    @model_validator(mode='after')
    def fill_logger(self):
        if not self.logger:
            self.logger = logger.get_logger('ActorsInLocaltionsPolar', level=self.verbosity)
        return self

    @model_validator(mode='after')
    def validate_max_transitions(self):
        if self.max_transitions < self.min_transitions:
            raise ValueError("max_transitions should be greater or equal than min_transitions")
        # it should be lower than the numbers of actors
        if self.max_transitions > len(self.entities.actors):
            raise ValueError("max_transitions should be lower than the number of actors")
        return self

    def create_ontology(self):
        """
        Creates an ontology based on the current state of entities and their locations.
        This method initializes and updates the states of entities (actors) in various locations
        based on the provided answer. The states are created and modified according to the following rules:
        - If `answer` is 1: The actor `a` is in location `l`.
        - If `answer` is 0: The actor `a` is not in location `l` and not in `NW`.
        - If `answer` is 2: Randomly decides between two conditions:
            - The actor `a` is in `NW` from the beginning.
            - The actor `a` is in location `l` at step i, and then moved to `NW`.
        """

        a = self.entities.actors[0]
        l = self.entities.locations[0]
        self.entities.locations.append(NW)
        self.logger.info("Creating ActorsInLocaltionsPolar", answer=self.answer, a=a, l=l)
        states = [None] * self.states_qty

        # Answer 1: a is in l
        if self.answer == 1:
            i = self.states_qty-1
            condition = lambda x: (a,l) in x
            states[i] = self.initialize_state(i, condition) # NewActorInLocationState
            for j in list(reversed(range(i))):
                any_condition = lambda x: True
                all_condition = lambda x: True
                states[j] = self.create_new_state(j, states[j+1], any_condition, all_condition)

        # Answer 0: a is not in l
        if self.answer == 0:
            i = self.states_qty-1
            condition = lambda x: (a, l) not in x and (a, NW) not in x
            states[i] = self.initialize_state(i, condition) # NewActorInLocationState
            for j in list(reversed(range(i))):
                any_condition = lambda x: True
                all_condition = lambda x: True
                states[j] = self.create_new_state(j, states[j+1], any_condition, all_condition)

        if self.answer == 2:
            if random.choice([0,1]):
                i = 0
                condition = lambda x: (a, NW) in x
                states[i] = self.initialize_state(i, condition)
                for j in range(1, self.states_qty):
                    any_condition = lambda pair : True
                    all_condition = lambda pair: pair[0] != a # all actros should be different from a, due it will be always in NW
                    states[j] = self.create_new_state(j, states[j-1], any_condition, all_condition)
            else:
                i = random.randint(0, self.states_qty-1)
                condition = lambda x: (a,l) in x
                states[i] = self.initialize_state(i, condition)
                for j in list(reversed(range(i))):
                    any_condition = lambda pair: True
                    all_condition = lambda pair: True
                    states[j] = self.create_new_state(j, states[j+1], any_condition, all_condition)

                # create the states after i
                for j in range(i+1, len(states)):
                    if j == i+1: # Here we place the actor in the NW.
                        any_condition = lambda pair: pair == (a, NW)
                        all_condition = lambda pair: True
                    else:
                        # Now there should be any ocurrence of actor
                        any_condition = lambda pair: True
                        all_condition = lambda pair: pair[0] != a
                    states[j] = self.create_new_state(j, states[j-1], any_condition, all_condition)

        self.states = states

    def initialize_state(self, i:int, condition:Callable)-> ActorInLocationState:
        """
        Initializes the state for an actor in a location based on a given condition.
        Args:
            i (int): An integer identifier for the state.
            condition (Callable): A callable that takes a set of entities and returns a boolean indicating 
                                  whether the condition is met.
        Returns:
            ActorInLocationState: The initialized state that meets the given condition.
        """

        self.logger.info("Creating Answer:", i=i)        
        s  = self.create_random_state(i)
        t = 0
        while not condition(s.entities_as_set):
            self.logger.debug("Condition not met", i=i, state=s)
            s  = self.create_random_state(i)
            t += 1
        self.logger.debug("State initialized", answer=self.answer, i=i)            
        return s

    def create_random_state(self, i:int)-> ActorInLocationState:
        """
        Creates a random state for actors in locations.
        Args:
            i (int): The index to be assigned to the generated state.
        Returns:
            ActorInLocationState: A state object containing actors and their randomly assigned locations.
        """


        actors = self.entities.actors
        locations = [random.choice(self.entities.locations) for _ in actors]
        actor_location_pairs = list(zip(actors, locations))
        e = []
        for pair in actor_location_pairs:
            e_i = ActorInLocationEntityState(actor=pair[0], location=pair[1])
            e.append(e_i)
        s = ActorInLocationState(entities=e, index=i)
        return s
    
    def create_new_state(self, j:int, state:ActorInLocationEntityState, any_condition:Callable, all_condition:Callable)-> ActorInLocationState:
        """
        Create a new state for an actor in a location based on the current state and the given conditions.
        Args:
            j (int): An identifier for the state.
            state (ActorInLocationEntityState): The current state of the actor in the location.
            any_condition (Callable): A callable that represents a condition to meet for any transition.
            all_condition (Callable): A callable that represents a condition to meet for all transitions.
        Returns:
            ActorInLocationState: The new state of the actor in the location after applying the transitions.
        """

        
        num_transitions = random.randint(self.min_transitions, self.max_transitions)
        delta = state.create_delta(num_transitions, self.entities.locations, any_condition, all_condition)
        self.logger.debug("Delta", i=j, delta=delta)
        new_state = state.create_state_from_delta(j, delta)
        return new_state
    
    def create_fol(self):
        sentences = []
        def enumerate_entities(state:ActorInLocationState):
            enumeration = []
            for e in state.entities:
                enumeration.append(["Exists", e.actor])
            return enumeration

        sentences.append(enumerate_entities(self.states[0]))           

        def describe_states(state:ActorInLocationState, prev_state:ActorInLocationState):
            state_sentences = []
            for entity in state.entities:
                if entity.location != NW:
                    # get the previous location for handling the FromTo case
                    prev_location = prev_state.get_actor_location(entity.actor)
                    state_sentences.append(random.choice([
                        ["In", entity.actor, entity.location],
                        ["To", entity.actor, entity.location],
                        ["FromTo", entity.actor, prev_location, entity.location]
                    ]))
            return state_sentences

        for i in range(1, self.states_qty):
            sentences.append(describe_states(self.states[i], self.states[i-1]))

        self.fol = sentences