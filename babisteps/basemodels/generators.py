import logging
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field, model_validator
from sparse import DOK, SparseArray

from babisteps import logger
from babisteps.basemodels.FOL import FOL, Exists, From, FromTo, In, IsRelated, To
from babisteps.basemodels.nodes import (
    Coordenate,
    Entity,
    EntityInCoordenateState,
    ImmediateGraph,
    Relationship,
    State,
)
from babisteps.basemodels.stories import Story

DELIM = "_-_"
ACTORS_NONE_ANSWERS = ["nobody", "no one"]
OBJECTS_LOCATION_EVENT_NONE_ANSWERS = ["nothing"]
UNKNONW_ANSWERS = [
    "unknown",
    "it is uncertain",
    "it is impossible to know",
    "not enough information",
    "it's impossible to know",
    "don't know",
]


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
                log_file=self.log_file,
            )
        return self

    def recreate(self):
        """Recreates the instance with the original input values."""
        self.logger.info(
            "Recreating instance with original inputs.",
            original_inputs=self.original_inputs,
        )
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


class OrderRequest(BaseModel, ABC):
    """
    Base class to be used in Temporal, Size and Spatial order.
    """

    answer: Any
    # Entities
    e0: Optional[Any] = None
    e1: Optional[Any] = None
    # Relation
    r: Optional[Any] = None
    relation_type: Literal["relative_event", "relative_size",
                           "relative_position", "absolute_position"]
    shape_str: Literal[("locations", ), ("actors", ), ("objects", ),
                       ("events", )]

    @abstractmethod
    def get_question(self):
        """Abstract method to generate the strings' question"""
        pass

    @abstractmethod
    def get_answer(self):
        """Abstract method to generate the answer"""
        pass


class OrderRequestPolar(OrderRequest):
    answer: Literal["yes", "no", "unknown"]

    def get_question(self) -> str:
        if self.shape_str in [("locations", ), ("objects", )]:
            return (f"Is the {self.e0.name} {random.choice(self.r.base)} "
                    f"the {self.e1.name}?")
        elif self.shape_str == ("actors", ):
            return f"Is {self.e0.name} {random.choice(self.r.base)} {self.e1.name}?"
        elif self.shape_str == ("events", ):
            return (f"Was the {self.e0.name} {random.choice(self.r.base)} "
                    f"the {self.e1.name}?")
        else:
            raise ValueError("Invalid shape_str for OrderRequestPolar")

    def get_answer(self) -> list[str]:
        if self.answer == "yes" or self.answer == "no":
            return [self.answer]
        elif self.answer == "unknown":
            return UNKNONW_ANSWERS
        else:
            raise ValueError("'answer' must be 'yes', 'no', or 'unknown'")


class OrderRequestHow(OrderRequest):
    answer: Literal["designated_relation", "unknown"]

    def get_question(self) -> str:
        if self.shape_str in [("locations", ), ("objects", )]:
            return f"How is the {self.e1.name} related to the {self.e0.name}?"
        elif self.shape_str == ("actors", ):
            return f"How is {self.e1.name} related to {self.e0.name}?"
        elif self.shape_str == ("events", ):
            return f"How was the {self.e1.name} related to the {self.e0.name}?"
        else:
            raise ValueError("Invalid shape_str for OrderRequestHow")

    def _get_answer_from_relation(self, relation: Relationship) -> list[str]:
        if self.answer == "designated_relation":
            answers = []
            if self.shape_str in [("locations", ), ("objects", )]:
                for i in relation.base:
                    answers.append(
                        f"the {self.e0.name} is {i} the {self.e1.name}")
                for j in relation.opposite:
                    answers.append(
                        f"the {self.e1.name} is {j} the {self.e0.name}")

            elif self.shape_str == ("actors", ):
                for i in relation.base:
                    answers.append(f"{self.e0.name} is {i} {self.e1.name}")
                for j in relation.opposite:
                    answers.append(f"{self.e1.name} is {j} {self.e0.name}")

            elif self.shape_str == ("events", ):
                for i in relation.base:
                    answers.append(
                        f"the {self.e0.name} was {i} the {self.e1.name}")
                for j in relation.opposite:
                    answers.append(
                        f"the {self.e1.name} was {j} the {self.e0.name}")
            else:
                raise ValueError("Invalid shape_str for OrderRequestHow")
            return answers

        elif self.answer == "unknown":
            return UNKNONW_ANSWERS
        else:
            raise ValueError(
                "'answer' must be 'designated_relation' or 'unknown'")

    def get_answer(self) -> list[str]:
        try:
            answer = self._get_answer_from_relation(self.r)
            return answer
        except Exception as e:
            raise e

    def get_options(self, relations: list[Relationship]) -> list[str]:
        """
        Extracts and returns a shuffled list of options from the provided relationships.
        """
        # keep current value of answer to then return it to its original value
        original_answer = self.answer
        # set answer to 'designated_relation' to produce all the options
        self.answer = "designated_relation"
        options = []
        for relation in relations:
            options.extend(self._get_answer_from_relation(relation))
        # now set the answer back to its original value
        self.answer = original_answer
        return options


class OrderRequestWhat(OrderRequest):
    answer: Literal["second_entity", "none", "unknown"]

    def get_question(self):
        if self.shape_str in [("locations", ), ("objects", )]:
            return f"To what is the {self.e0.name} {random.choice(self.r.base)}?"
        elif self.shape_str == ("actors", ):
            return f"To who is {self.e0.name} {random.choice(self.r.base)}?"
        elif self.shape_str == ("events", ):
            return f"To what was the {self.e0.name} {random.choice(self.r.base)}?"
        else:
            raise ValueError("Invalid shape_str for OrderRequestWhat")

    def get_answer(self):
        if self.answer == "second_entity":
            return [self.e1.name]
        elif self.answer == "none":
            if self.shape_str == ("actors", ):
                return ACTORS_NONE_ANSWERS
            else:
                return OBJECTS_LOCATION_EVENT_NONE_ANSWERS
        elif self.answer == "unknown":
            return UNKNONW_ANSWERS
        else:
            raise ValueError(
                "'answer' must be 'second_entity', 'none', or 'unknown'")


class OrderModel(BaseModel):
    entities: list[Entity]
    relations: list[Relationship]

    @model_validator(mode="after")
    def _shuffle(self):
        random.shuffle(self.entities)
        return self

    @property
    def as_tuple(self):
        return (self.entities, )


class OrderBaseGenerator(BaseGenerator):
    model: OrderModel
    edge_qty: int
    graphs: Optional[list[ImmediateGraph]] = None
    topic: Any
    # shape is a tuple of multiple integers
    shape: Optional[tuple[int]] = None
    shape_str: Optional[tuple[str]] = None
    story: Optional[Story] = None
    fol: list[FOL] = None
    nl: list[str] = None

    @model_validator(mode="before")
    def _validate_edge_qty(cls, values):
        # 0 < edge_qty < ((n^2)-n)/2
        n = len(values["model"].entities)
        if values["edge_qty"] < 0 or values["edge_qty"] > ((n**2) - n) / 2:
            raise ValueError("edge_qty must be between 0 and ((n^2)-n)/2")
        return values

    @abstractmethod
    def load_ontology_from_topic(self) -> Callable:
        """Abstract method to be implemented in subclasses."""
        pass

    def create_ontology(self):
        f_ontology = self.load_ontology_from_topic()
        self.graphs = f_ontology()

    def _transitive_closure(self, matrix: SparseArray, g: nx.DiGraph):
        # TODO: Add description
        TC = nx.transitive_closure(g, reflexive=None)
        for i, j in TC.edges:
            matrix[i, j] = 1
            matrix[j, i] = 0
        for i, j in matrix.data:
            if matrix[i, j] == 1:
                g.add_edge(i, j)
        return matrix, g

    def _fill_edges(
        self,
        matrix: SparseArray,
        g: nx.DiGraph,
        n: int,
    ):
        # TODO: Add description
        nans = np.isnan(matrix.todense())
        origin_mask = ~nans
        origin_vals = matrix.todense()[origin_mask]
        nans = np.argwhere(nans)
        while len(nans) > 0:
            c = np.random.choice(len(nans))
            c = nans[c]
            i, j = int(c[0]), int(c[1])
            matrix_aux = deepcopy(matrix)
            g_aux = deepcopy(g)
            matrix_aux[i, j] = 1
            matrix_aux[j, i] = 0
            g_aux.add_edge(i, j)
            matrix_aux, g_aux = self._transitive_closure(matrix_aux, g_aux)
            # validate that origin_vals are not changed
            new_vals = matrix_aux.todense()[origin_mask]
            if not np.array_equal(origin_vals, new_vals):
                matrix[i, j] = 0
                continue
            if len(g_aux.edges) == n:
                return matrix_aux, g_aux
            elif len(g_aux.edges) < n:
                matrix = matrix_aux
                g = g_aux
            elif len(g_aux.edges) > n:
                matrix[i, j] = 0
            else:
                raise ValueError("This should not happen")
            nans = np.argwhere(np.isnan(matrix.todense()))

    def _create_aux(self):
        self.shape = (len(self.model.relations), len(self.model.entities))
        self.shape_str = self.topic.shape_str

    def _create_empty_graph(self) -> tuple[DOK, nx.DiGraph]:
        """
        Create an empty graph represented as a Sparse matrix and a DiGraph
        """
        g_am = DOK((self.shape[1], self.shape[1]), fill_value=np.nan)
        g = nx.DiGraph()
        for i, e in enumerate(self.model.entities):
            g_am[i, i] = 0  # diagonal is 0
            g.add_node(i, entity=e)
        return g_am, g

    def _init_setup(self) -> tuple[Entity, Entity, SparseArray, nx.DiGraph]:
        """
        Setup the initial args of our generator, and
        fill the elements in the topic/scenario.
        """
        e0 = self.model.entities[0]
        e1 = self.model.entities[1]
        self._create_aux()
        self.topic.e0 = e0
        self.topic.e1 = e1
        self.topic.r = self.model.relations[0]
        r_am, r = self._create_empty_graph()

        return e0, e1, r_am, r

    def create_fol(self):

        def enumerate_model(element: list, shape_type: str) -> list[list]:
            enumeration = []
            for e in element:
                enumeration.append(Exists(thing=e, shape_str=shape_type))
            return enumeration

        def describe_relation(relation, graph, shape_str):
            graph_sentences = []
            for i, j in graph.am.data:
                # check first am
                edge = graph.am[i, j]
                if edge == 1:
                    # then verify if the edge exists in the graph
                    assert graph.g.has_edge(
                        i,
                        j), ("edge {}-{} does not exist in the graph".format(
                            i, j))
                    graph_sentences.append(
                        IsRelated(
                            relation=relation,
                            entity0=self.model.entities[i],
                            entity1=self.model.entities[j],
                            shape_str=shape_str,
                        ))
            return graph_sentences

        world_enumerate = []
        story = []
        # World enumeration
        for t, dim_str in zip(self.model.as_tuple, self.shape_str):
            world_enumerate.extend(enumerate_model(t, dim_str))
        random.shuffle(world_enumerate)
        # Story
        for relation, graph in zip(self.model.relations, self.graphs):
            story.extend(describe_relation(relation, graph, self.shape_str))
        random.shuffle(story)

        self.story = Story(
            world_enumerate=world_enumerate,
            describe_len=0,
            story=story,
            question=self.topic.get_question(),
            answer=self.topic.get_answer(),
        )
        # FOL
        self.fol = world_enumerate + story

    def create_nl(self):
        self.nl = [f.to_nl() for f in self.fol]

    def generate(self):
        self.create_ontology()
        self.create_fol()

    @abstractmethod
    def get_json(self, **kwargs):
        """Abstract method to be implemented in subclasses."""
        pass

    def get_txt(self):
        return self.story.create_txt()
