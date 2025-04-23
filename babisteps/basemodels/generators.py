import logging
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, model_validator
from sparse import DOK

from babisteps import logger
from babisteps.basemodels.FOL import FOL, Exists, From, FromTo, In, To
from babisteps.basemodels.nodes import (Coordenate, Entity,
                                        EntityInCoordenateState, State)
from babisteps.basemodels.stories import Story

DELIM = "_-_"


class BaseGenerator(BaseModel, ABC):
    verbosity: Union[int, str] = Field(default=logging.INFO)
    logger: Optional[Any] = None
    log_file: Optional[Path] = None
    original_inputs: Optional[dict] = None
    name: Optional[str] = None

    @model_validator(mode="before")
    def save_inputs_dict(cls, values):
        values["original_inputs"] = deepcopy(values)
        return values

    @model_validator(mode="after")
    def fill_logger(self):
        if not self.logger:
            self.logger = logger.get_logger(
                self.__class__.__name__ if self.name is None else
                self.__class__.__name__ + "-" + self.name,
                level=self.verbosity,
                log_file=self.log_file)
        return self

    def recreate(self):
        """Recreates the instance with the original input values."""
        self.logger.info("Recreating instance with original inputs.",
                         original_inputs=self.original_inputs)
        if self.original_inputs is None:
            raise ValueError("Original inputs not available.")
        # Use self.__class__ so that the child class is recreated
        return self.__class__(**self.original_inputs)

    @abstractmethod
    def generate(self, **kwargs):
        """Abstract method to be implemented in subclasses."""
        pass

    @abstractmethod
    def get_json(self, **kwargs):
        """Abstract method to be implemented in subclasses."""
        pass

    @abstractmethod
    def get_txt(self, **kwargs):
        """Abstract method to be implemented in subclasses."""
        pass


class SimpleTrackerBaseGenerator(BaseGenerator):
    model: Any
    states_qty: int
    topic: Any
    uncertainty: Optional[Coordenate] = None
    states: Optional[list[State]] = None
    deltas: Optional[Any] = None
    story: Optional[Story] = None
    fol: list[FOL] = None
    nl: list[str] = None
    num_transitions: int = 1
    idx2e: Optional[dict] = None
    e2idx: Optional[dict] = None
    idx2c: Optional[dict] = None
    c2idx: Optional[dict] = None
    shape: Optional[tuple[int, int]] = None
    shape_str: Literal[("locations", "actors"), ("actors", "objects")]

    @model_validator(mode="after")
    def check_shape_and_model(self):
        model_tuple = self.model.as_tuple
        if len(model_tuple) != len(self.shape_str):
            raise ValueError(
                f"Length mismatch: 'model.as_tuple()' has length {len(model_tuple)} "
                f"but 'shape_str' has length {len(self.shape_str)}.")
        return self

    @abstractmethod
    def load_ontology_from_topic(self) -> Callable:
        """Abstract method to be implemented in subclasses."""
        pass

    def _create_aux(self):
        self.shape = (len(self.model.coordenates), len(self.model.entities))
        self.idx2e = {i: e for i, e in enumerate(self.model.entities)}
        self.e2idx = {e: i for i, e in enumerate(self.model.entities)}
        self.idx2c = {i: c for i, c in enumerate(self.model.coordenates)}
        self.c2idx = {c: i for i, c in enumerate(self.model.coordenates)}
        return

    def create_ontology(self):
        f_ontology = self.load_ontology_from_topic()
        self.states = f_ontology()
        self.create_transitions()

    def create_new_state(
        self,
        j: int,
        state: EntityInCoordenateState,
        condition: Callable,
    ) -> EntityInCoordenateState:
        """
        Create a new state for an entity in a location based on the current state and
        the given conditions.
        Args:
            j (int): An identifier for the state.
            state (EntityInCoordenateState): The current state of derive a new one.
            condition (Callable): A callable that represents a condition to meet by
            the transition.
        Returns:
            EntityInCoordenateState: The new state of the entity in the location after
            applying the transitions.
        """

        new_am, _ = state.create_transition(
            self.num_transitions,
            condition,
        )
        new_state = EntityInCoordenateState(am=new_am,
                                            index=j,
                                            verbosity=self.verbosity,
                                            log_file=self.log_file)
        return new_state

    def initialize_state(self, i: int,
                         condition: Callable) -> EntityInCoordenateState:
        """
        Initializes the state for an entity in a location based on a given condition.
        Args:
            i (int): An integer identifier for the state.
            condition (Callable): A callable that takes a set of entities and returns a
            boolean indicating
                                  whether the condition is met.
        Returns:
            EntityInCoordenateState: A initialized state that meets the given condition.
        """

        self.logger.info("Creating Answer:", i=i)
        s = self.create_random_state(i)
        t = 0
        while not condition(s.am):
            self.logger.debug("Condition not met", i=i, state=s)
            s = self.create_random_state(i)
            t += 1

        self.logger.debug("State initialized",
                          state=s,
                          answer=self.topic.answer,
                          i=i)
        return s

    def create_random_state(self, i: int) -> EntityInCoordenateState:
        """
        Creates a random state for entities in coordenates.
        Args:
            i (int): The index to be assigned to the generated state.
        Returns:
            EntityInCoordenateState: A state represented as an adjacency matrix,
            in sparse format (DOK).
        """

        entities = np.arange(self.shape[1])
        coordenates = np.random.choice(self.shape[0],
                                       self.shape[1],
                                       replace=True)
        sparse_matrix = DOK(shape=self.shape, dtype=int, fill_value=0)
        entity_coord_pairs = list(zip(coordenates, entities))
        for x, y in entity_coord_pairs:
            sparse_matrix[x, y] = 1
        s = EntityInCoordenateState(am=sparse_matrix,
                                    index=i,
                                    verbosity=self.verbosity,
                                    log_file=self.log_file)
        return s

    def create_transitions(self):
        deltas = []

        for i in range(0, self.states_qty - 1):
            current_state, reference_state = (
                self.states[i + 1].am,
                self.states[i].am,
            )

            diff = current_state.to_coo() - reference_state.to_coo()
            deltas_i = []
            for j in range(0, len(diff.data), 2):
                # get by pairs
                pair = diff.data[j:j + 2]
                if pair[0] == -1:
                    o = j
                    e = j + 1
                else:
                    o = j + 1
                    e = j
                delta_j = np.array([diff.coords.T[o], diff.coords.T[e]])
                self.logger.info("Transition", i=i, transition=delta_j)
                deltas_i.append(delta_j)
            deltas.append(deltas_i)
        self.deltas = deltas

    def create_fol(self):

        def enumerate_model(element: Union[list[Entity], list[Coordenate]],
                            shape_type: str) -> list[list]:
            enumeration = []
            for e in element:
                if e != self.uncertainty:
                    enumeration.append(Exists(thing=e, shape_str=shape_type))
            return enumeration

        def describe_states(state: State) -> list[list]:
            state_sentences = []
            for unit in state.am.data:
                x, y = unit[0], unit[1]
                e, c = self.idx2e[y], self.idx2c[x]
                if c != self.uncertainty:
                    state_sentences.append(
                        In(entity=e, coordenate=c, shape_str=self.shape_str))
            return state_sentences

        def describe_transitions(state: State) -> list[list]:
            i = state.index
            delta = self.deltas[i]
            transition_sentences = []
            for d in delta:
                idx_entity = d[0, 1]
                idx_prev_coord = d[0, 0]
                idx_next_coord = d[1, 0]
                entity = self.idx2e[idx_entity]
                prev_coord = self.idx2c[idx_prev_coord]
                next_coord = self.idx2c[idx_next_coord]
                if prev_coord == self.uncertainty:
                    transition_sentences.append(
                        To(
                            entity=entity,
                            coordenate=next_coord,
                            shape_str=self.shape_str,
                        ))
                elif next_coord == self.uncertainty:
                    transition_sentences.append(
                        From(
                            entity=entity,
                            coordenate=prev_coord,
                            shape_str=self.shape_str,
                        ))
                else:
                    transition_sentences.append(
                        random.choice([
                            To(
                                entity=entity,
                                coordenate=next_coord,
                                shape_str=self.shape_str,
                            ),
                            FromTo(
                                entity=entity,
                                coordenate1=prev_coord,
                                coordenate2=next_coord,
                                shape_str=self.shape_str,
                            ),
                        ]))
            return transition_sentences

        world_enumerate = []
        story = []

        for t, dim_str in zip(self.model.as_tuple, self.shape_str):
            world_enumerate.extend(enumerate_model(t, dim_str))
        random.shuffle(world_enumerate)
        story.extend(describe_states(self.states[0]))
        random.shuffle(story)
        describe_len = len(story)

        for s in self.states[0:-1]:
            story.extend(describe_transitions(s))

        self.story = Story(
            world_enumerate=world_enumerate,
            describe_len=describe_len,
            story=story,
            question=self.topic.get_question(),
            answer=self.topic.get_answer(),
        )
        self.fol = world_enumerate + story

    def create_nl(self):
        self.nl = [f.to_nl() for f in self.fol]

    def print_transition(self):
        self.logger.info("Initial state", state=self.states[0].am.todense())
        for i, d in enumerate(self.deltas):
            aux = [[x[0][0], x[0][1], x[1][1]] for x in d]
            for d in aux:
                self.logger.info("Delta", i=i, e=d[0], prev=d[1], next=d[2])
        self.logger.info("Final state", state=self.states[0].am.todense())
