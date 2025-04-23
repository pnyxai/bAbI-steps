import random
from copy import deepcopy
from typing import Any, Callable, Literal, Optional, get_type_hints

import networkx as nx
import numpy as np
from pydantic import BaseModel, model_validator
from sparse import DOK, SparseArray

from babisteps.basemodels.FOL import FOL, Exists, IsRelated
from babisteps.basemodels.generators import DELIM, BaseGenerator
from babisteps.basemodels.nodes import Entity, ImmediateGraph, Relationship
from babisteps.basemodels.stories import Story


class ImmediateOrderRequest(BaseModel):
    answer: Any
    # Entity
    e0: Optional[Any] = None
    e1: Optional[Any] = None
    # Relation
    r: Optional[Any] = None
    relation_type: Literal["relative_event", "relative_size",
                           "relative_position", "absolute_position"]
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
        return [self.answer]


class ImmediateOrderRequestHow(ImmediateOrderRequest):
    answer: Literal["designated_relation", "unknown"]

    def get_question(self):
        if self.shape_str in [("locations", ), ("objects", )]:
            return f"How is the {self.e1.name} related to the {self.e0.name}?"
        elif self.shape_str == ("actors", ):
            return f"How is {self.e1.name} related to {self.e0.name}?"
        elif self.shape_str == ("events", ):
            return f"How was the {self.e1.name} related to the {self.e0.name}?"
        else:
            raise ValueError("Invalid shape_str for ImmediateOrderRequestHow")

    def _get_answer_from_relation(self, relation: Relationship):
        if self.answer == "designated_relation":
            answers = []
            if self.shape_str in [("locations", ), ("objects", )]:
                for i in relation.base:
                    answers.append(
                        f"The {self.e0.name} is {i} the {self.e1.name}")
                for j in relation.opposite:
                    answers.append(
                        f"The {self.e1.name} is {j} the {self.e0.name}")

            elif self.shape_str == ("actors", ):
                for i in relation.base:
                    answers.append(f"{self.e0.name} is {i} {self.e1.name}")
                for j in relation.opposite:
                    answers.append(f"{self.e1.name} is {j} {self.e0.name}")

            elif self.shape_str == ("events", ):
                for i in relation.base:
                    answers.append(
                        f"The {self.e0.name} was {i} the {self.e1.name}")
                for j in relation.opposite:
                    answers.append(
                        f"The {self.e1.name} was {j} the {self.e0.name}")
            else:
                raise ValueError(
                    "Invalid shape_str for ImmediateOrderRequestHow")
            return answers

        elif self.answer == "unknown":
            return [self.answer]
        else:
            raise ValueError(
                "'answer' must be 'designated_relation' or 'unknown'")

    def get_answer(self):
        try:
            answer = self._get_answer_from_relation(self.r)
            return answer
        except Exception as e:
            raise e

    def get_options(self, relations: list[Relationship]):
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


class ImmediateOrderRequestWhat(ImmediateOrderRequest):
    answer: Literal["second_entity", "none", "unknown"]

    def get_question(self):
        if self.shape_str in [("locations", ), ("objects", )]:
            return f"To what is the {self.e0.name} {random.choice(self.r.base)}?"
        elif self.shape_str == ("actors", ):
            return f"To who is {self.e0.name} {random.choice(self.r.base)}?"
        elif self.shape_str == ("events", ):
            return f"To what was the {self.e0.name} {random.choice(self.r.base)}?"
        else:
            raise ValueError("Invalid shape_str for ImmediateOrderRequestWhat")

    def get_answer(self):
        if self.answer == "second_entity":
            return [self.e1.name]
        elif self.answer == "none" or self.answer == "unknown":
            return [self.answer]
        else:
            raise ValueError(
                "'answer' must be 'second_entity', 'none', or 'unknown'")


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
            ImmediateOrderRequestWhat: self._immediate_order_what,
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

    def _fill_edges(
        self,
        matrix: SparseArray,
        g: nx.DiGraph,
        n: int,
    ):
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

        self.logger.debug(
            "Creating Immediate Order Graph",
            answer=self.topic.answer,
            e0=e0.name,
            e1=e1.name,
            relation=self.topic.r.name,
        )
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
        self.topic.r = self.model.relations[0]
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
                g_am[0, 1], g_am[1, 0] = 0, 0
                g_am, g = self._fill_edges(g_am, g, self.edge_qty)
                graphs.append(
                    ImmediateGraph(am=g_am, g=g, name=rlt.name, index=i))
        return graphs

    def _immediate_order_what(self):
        e0 = self.model.entities[0]
        e1 = self.model.entities[1]
        graphs = []
        self._create_aux()
        self.topic.e0 = e0
        self.topic.e1 = e1
        self.topic.r = self.model.relations[0]
        r_am, r = self._create_empty_graph()
        if self.topic.answer == "second_entity":
            r_am[0, 1] = 1
            r_am[1, 0] = 0
            r.add_edge(0, 1)
            r_am[0, 2:] = 0
        elif self.topic.answer == "none":
            if self.edge_qty <= len(self.model.entities) - 1:
                self.logger.error(
                    "The #edges must be <= that # entities when the answer is 'none'",
                    answer=self.topic.answer,
                    edge_qty=self.edge_qty,
                    n_entities=len(self.model.entities),
                )
                raise ValueError(
                    "The #edges must be <= that # entities when the answer is 'none'"
                )
            r_am[0, :] = 0
            r_am[1:, 0] = 1
            for i in range(1, len(self.model.entities)):
                r.add_edge(i, 0)
        elif self.topic.answer == "unknown":
            rnd_e = random.choice(range(1, len(self.model.entities)))
            r_am[rnd_e, 0] = 0
            r_am[0, 1:] = 0
        else:
            raise ValueError(
                "Invalid answer should be 'second_entity' or 'unknown'")
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
        options = list(get_type_hints(self.topic)["answer"].__args__)
        if isinstance(self.topic, ImmediateOrderRequestPolar):
            pass
        elif isinstance(self.topic, ImmediateOrderRequestHow):
            options.remove("designated_relation")
            o = self.topic.get_options(self.model.relations)
            options.extend(o)
        elif isinstance(self.topic, ImmediateOrderRequestWhat):
            options.remove("second_entity")
            options.extend([e.name for e in self.model.entities])

        random.shuffle(options)
        json["options"] = options
        if self.name and DELIM in self.name:
            parts = self.name.split(DELIM)
            if len(parts) == 3:
                json["leaf"] = parts[0]
                json["leaf_label"] = parts[1]
                json["leaf_index"] = parts[2]
            else:
                raise ValueError(
                    "self.name does not contain exactly three parts separated by '_-_'"
                )
        else:
            raise ValueError(
                "self.name is either None or does not contain the delimiter '_-_'"
            )

        return json

    def get_txt(self):
        return self.story.create_txt()
