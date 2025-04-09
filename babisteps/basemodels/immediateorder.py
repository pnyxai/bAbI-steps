import random
from copy import deepcopy
from typing import Any, Callable, Literal, Optional, get_type_hints

import networkx as nx
import numpy as np
from pydantic import BaseModel, model_validator
from sparse._dok import DOK
from sparse._sparse_array import SparseArray

from babisteps.basemodels.FOL import FOL, Exists, IsRelated
from babisteps.basemodels.generators import BaseGenerator
from babisteps.basemodels.nodes import Entity, ImmediateGraph, Relationship
from babisteps.basemodels.stories import Story


class ImmediateOrderRequest(BaseModel):
    answer: Any
    # Entity
    e0: Optional[Any] = None
    e1: Optional[Any] = None
    # Relation
    r: Optional[Any] = None
    relation_type: Literal['relative_event', 'relative_size',
                           'relative_position', 'absolute_position']
    shape_str: Literal[("locations", ), ("actors", ), ("objects", ),
                       ("events", )]

    def get_question(self):
        pass

    def get_answer(self):
        pass


class ImmediateOrderRequestPolar(ImmediateOrderRequest):
    answer: Literal["yes", "no", "unknown"]

    def get_question(self):

        if self.shape_str in [("locations", ), ("objects", )]:
            return (f"Is the {self.e0.name} {random.choice(self.r.base)} "
                    f"the {self.e1.name}?")
        elif self.shape_str == ("actors", ):
            return f"Is {self.e0.name} {random.choice(self.r.base)} {self.e1.name}?"
        elif self.shape_str == ("events", ):
            return (f"Was the {self.e0.name} {random.choice(self.r.base)} "
                    f"the {self.e1.name}?")
        else:
            raise ValueError(
                "Invalid shape_str for ImmediateOrderRequestPolar")

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

    def _fill_edges(self,
                    matrix: SparseArray,
                    g: nx.DiGraph,
                    n: int,
                    condition: Optional[Callable] = None):
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
            # if condition is not none, check it
            if condition and not condition(matrix_aux, g_aux):
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

    def _create_empty_graph(self):
        g_am = DOK((self.shape[1], self.shape[1]), fill_value=np.nan)
        g = nx.DiGraph()
        for i, e in enumerate(self.model.entities):
            g_am[i, i] = 0  # diagonal is 0
            g.add_node(i, entity=e)
        return g_am, g

    def _immediate_order_polar(self):
        e0 = self.model.entities[0]
        e1 = self.model.entities[1]
        graphs = []
        self._create_aux()
        self.topic.e0 = e0
        self.topic.e1 = e1
        self.topic.r = self.model.relations[0]
        r_am, r = self._create_empty_graph()
        condition = None  # Re-defined only in the unknown case
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
            # validate that both 0,1 and 1,0 are not in the graph
            #condition = lambda r_am, r: (r_am[0, 1] == 0 and r_am[1, 0] == 0) and /
            # (not r.has_edge(0, 1) and not r.has_edge(1, 0))
            condition = lambda r_am, r: not r.has_edge(
                0, 1) and not r.has_edge(1, 0)

        self.logger.debug("Creating Immediate Order Graph",
                          answer=self.topic.answer,
                          e0=e0.name,
                          e1=e1.name,
                          relation=self.topic.r.name)
        r_am, r = self._fill_edges(r_am, r, self.edge_qty, condition)
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

    def create_fol(self):

        def enumerate_model(element: list, shape_type: str) -> list[list]:
            enumeration = []
            for e in element:
                enumeration.append(Exists(thing=e, shape_str=shape_type))
            return enumeration

        def describe_relation(relation, graph, shape_str):
            graph_sentences = []
            for (i, j) in graph.am.data:
                # check first am
                edge = graph.am[i, j]
                if edge == 1:
                    # then verify if the edge exists in the graph
                    assert graph.g.has_edge(
                        i, j), "edge {}-{} does not exist in the graph".format(
                            i, j)
                    graph_sentences.append(
                        IsRelated(relation=relation,
                                  entity0=self.model.entities[i],
                                  entity1=self.model.entities[j],
                                  shape_str=shape_str))
            return graph_sentences

        world_enumerate = []
        story = []

        for t, dim_str in zip(self.model.as_tuple, self.shape_str):

            world_enumerate.extend(enumerate_model(t, dim_str))
        random.shuffle(world_enumerate)

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

        self.fol = world_enumerate + story

    def create_nl(self):
        self.nl = [f.to_nl() for f in self.fol]

    def generate(self):
        self.create_ontology()
        self.create_fol()

    def get_json(self):
        json = self.story.create_json()
        options = list(get_type_hints(self.topic)['answer'].__args__)
        if isinstance(self.topic, ImmediateOrderRequestPolar):
            # do nothing
            pass
        elif isinstance(self.topic, ImmediateOrderRequestHow):
            options.remove('designated_relation')
            options.extend([r.name for r in self.model.relations])
        elif isinstance(self.topic, ImmediateOrderRequestWhat):
            options.remove('second_designated_event')
            options.extend([e.name for e in self.model.entities])

        random.shuffle(options)
        json['options'] = options
        if self.name:
            json['leaf'] = self.name.split('_-_')[0]
            json['leaf_label'] = self.name.split('_-_')[1]
            json['leaf_index'] = self.name.split('_-_')[2]
        return json

    def get_txt(self):
        return self.story.create_txt()
