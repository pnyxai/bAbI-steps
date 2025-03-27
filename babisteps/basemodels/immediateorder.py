import random
from copy import deepcopy
from typing import Any, Callable, Literal, Optional

import networkx as nx
import numpy as np
from pydantic import BaseModel, model_validator
from sparse._dok import DOK
from sparse._sparse_array import SparseArray

from babisteps.basemodels.generators import BaseGenerator
from babisteps.basemodels.nodes import Entity, ImmediateGraph, Relationship


class ImmediateOrderRequest(BaseModel):
    answer: Any
    # Event
    e0: Optional[Any] = None
    e1: Optional[Any] = None
    # Relation
    r0: Optional[Any] = None
    r1: Optional[Any] = None
    relation_type: Literal["relative_event", "relative_size",
                           "relative_position", "absolute_position"]

    def get_question(self):
        pass

    def get_answer(self):
        pass


class ImmediateOrderRequestPolar(ImmediateOrderRequest):
    answer: Literal["yes", "no", "unknown"]

    def get_question(self):
        return f"Is {self.e0.name} {self.r0} {self.e1.name}?"

    def get_answer(self):
        return self.answer


class ImmediateOrderRequestHow(ImmediateOrderRequest):
    answer: Literal["designated_relation", "unknown"]

    def get_question(self):
        return f"How is {self.e1.name} related to {self.e0.name}?"

    def get_answer(self):
        # TODO: Probably the way to get the answer should be different
        # and depend on the relation type?
        if self.answer == "designated_relation":
            return self.r0
        elif self.answer == "unknown":
            return self.answer
        else:
            raise ValueError(
                "'answer' must be 'designated_relation' or 'unknown'")


class ImmediateOrderRequestWhat(ImmediateOrderRequest):
    answer: Literal["second_designated_event", "unknown"]

    def get_question(self):
        # TODO: The question generation is related to the relation type
        # probably this would need to be handle differently.
        return f"To what is {self.e0.name} {self.r0}?"

    def get_answer(self):
        if self.answer == "second_designated_event":
            return self.e1.name
        elif self.answer == "unknown":
            return self.answer
        else:
            raise ValueError(
                "'answer' must be 'second_designated_event' or 'unknown'")


class ImmediateOrderModel(BaseModel):
    entities: list[Entity]
    relations: list[Relationship]

    @model_validator(mode="after")
    def _shuffle(self):
        random.shuffle(self.entities)
        return self

    @property
    def as_tuple(self):
        return (self.entities, )


class ImmediateOrder(BaseGenerator):
    model: Any
    edge_qty: int
    graphs: Optional[list[ImmediateGraph]] = None
    topic: ImmediateOrderRequest
    # shape is a tuple of multiple integers
    shape: Optional[tuple[int]] = None

    @model_validator(mode="before")
    def _validate_edge_qty(cls, values):
        # 0 < edge_qty < ((n^2)-n)/2
        n = len(values["model"].entities)
        if values["edge_qty"] < 0 or values["edge_qty"] > ((n**2) - n) / 2:
            raise ValueError("edge_qty must be between 0 and ((n^2)-n)/2")
        return values

    def load_ontology_from_topic(self) -> Callable:
        # Define the mapping between answer types and loader functions
        loader_mapping: dict[type[ImmediateOrderRequest], Callable] = {
            ImmediateOrderRequestPolar: self._immediate_order_polar,
            ImmediateOrderRequestHow: self._immediate_order_how,
        }
        # Get the type of the answer
        topic_type = type(self.topic)

        return loader_mapping[topic_type]

    def create_ontology(self):
        f_ontology = self.load_ontology_from_topic()
        self.graphs = f_ontology()

    def _transitive_closure(self, matrix: SparseArray, g: nx.DiGraph):
        TC = nx.transitive_closure(g, reflexive=None)
        for i, j in TC.edges:
            matrix[i, j] = 1
            matrix[j, i] = 0
        for i, j in matrix.data:
            if matrix[i, j] == 1:
                g.add_edge(i, j)
        return matrix, g

    def _fill_edges(self, matrix: SparseArray, g: nx.DiGraph, n: int):
        nans = np.argwhere(np.isnan(matrix.todense()))
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

    def _create_empty_graph(self):
        g_am = DOK((self.shape[1], self.shape[1]), fill_value=np.nan)
        g = nx.DiGraph()
        for i, e in enumerate(self.model.entities):
            g_am[i, i] = 0
            g.add_node(i, entity=e)
        return g_am, g

    def _immediate_order_polar(self):
        e0 = self.model.entities[0]
        e1 = self.model.entities[1]
        graphs = []
        self._create_aux()
        self.topic.e0 = e0
        self.topic.e1 = e1
        r_am, r = self._create_empty_graph()
        if self.topic.answer == "yes":
            r_am[0, 1] = 1
            r_am[1, 0] = 0
            r.add_edge(0, 1)
        elif self.topic.answer == "no":
            r_am[0, 1] = 0
            r_am[1, 0] = 1
            r.add_edge(1, 0)
        elif self.topic.answer == "unknown":
            r_am[0, 1] = 0
            r_am[1, 0] = 0
        r_am, r = self._fill_edges(r_am, r, self.edge_qty)
        graphs.append(
            ImmediateGraph(am=r_am,
                           g=r,
                           name=self.model.relations[0].name,
                           index=0))
        if len(self.model.relations) > 1:
            for i, rlt in enumerate(self.model.relations[1:], start=1):
                g_am, g = self._create_empty_graph()
                g_am, g = self._fill_edges(g_am, g, self.edge_qty)
                graphs.append(
                    ImmediateGraph(am=g_am, g=g, name=rlt.name, index=i))
        return graphs

    def _immediate_order_how(self):
        e0 = self.model.entities[0]
        e1 = self.model.entities[1]
        graphs = []
        self._create_aux()
        self.topic.e0 = e0
        self.topic.e1 = e1
        r_am, r = self._create_empty_graph()
        if self.topic.answer == "designated_relation":
            r_am[0, 1] = 1
            r_am[1, 0] = 0
            r.add_edge(0, 1)
        elif self.topic.answer == "unknown":
            r_am[0, 1] = 0
            r_am[1, 0] = 0
        else:
            raise ValueError(
                "Invalid answer should be 'designated_relation' or 'unknown'")
        r_am, r = self._fill_edges(r_am, r, self.edge_qty)
        graphs.append(
            ImmediateGraph(am=r_am,
                           g=r,
                           name=self.model.relations[0].name,
                           index=0))
        if len(self.model.relations) > 1:
            for i, rlt in enumerate(self.model.relations[1:], start=1):
                g_am, g = self._create_empty_graph()
                if (self.topic.answer == "designated_relation"
                        or self.topic.answer == "unknown"):
                    g_am[0, 1] = 0
                    g_am[1, 0] = 0
                g_am, g = self._fill_edges(g_am, g, self.edge_qty)
                graphs.append(
                    ImmediateGraph(am=g_am, g=g, name=rlt.name, index=i))
        return graphs

    def generate(self):
        self.create_ontology()

    def get_json(self):
        return

    def get_txt(self):
        return
