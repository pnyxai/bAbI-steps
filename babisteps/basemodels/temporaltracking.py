import random
from abc import abstractmethod
from copy import deepcopy
from itertools import groupby
from typing import Any, Callable, Literal, Optional

import networkx as nx
import numpy as np
from pydantic import BaseModel, model_validator
from sparse import DOK, SparseArray

from babisteps.basemodels.FOL import Exists, IsTemporalRelated
from babisteps.basemodels.generators import (
    ACTORS_NONE_ANSWERS,
    DELIM,
    REPLACE_PLACEHOLDER,
    UNKNONW_ANSWERS,
    OrderBaseGenerator,
)
from babisteps.basemodels.nodes import Coordinate, Entity, Relationship, TemporalTrackingEvent, TemporalTrackingGraph
from babisteps.basemodels.simpletracking import EntitiesInCoordinates
from babisteps.basemodels.stories import Story

MAX_ATTEMPTS_MULTIPLIER = 10


class TemporalTrackingRequest(BaseModel):
    answer: Any
    # Events
    e0: Optional[TemporalTrackingEvent] = None
    e1: Optional[TemporalTrackingEvent] = None
    # Relation
    r: Relationship
    relation_type: Literal["relative_event"]
    shape_str: tuple[str, str] = ("locations", "actors")

    @abstractmethod
    def get_question(self):
        """Abstract method to generate the strings' question"""
        pass

    @abstractmethod
    def get_answer(self):
        """Abstract method to generate the answer"""
        pass

    @abstractmethod
    def get_response_template(self):
        """Abstract method to generate the answer context template"""
        pass


class TemporalTrackingPolarRequest(TemporalTrackingRequest):
    answer: Literal["yes", "no", "unknown"]

    def get_question(self):
        question_options = [
            "Did {e0} go to the {l0} {r_base} {e1} went to the {l1}?",
            "Did {e1} go to the {l1} {r_opposite} {e0} went to the {l0}?",
        ]
        question_template = random.choice(question_options)
        e0, e1 = self.e0.entity.name, self.e1.entity.name
        l0, l1 = self.e0.coordinate.name, self.e1.coordinate.name
        r_base, r_opposite = random.choice(self.r.base), random.choice(self.r.opposite)
        return question_template.format(e0=e0, e1=e1, l0=l0, l1=l1, r_base=r_base, r_opposite=r_opposite)

    def get_answer(self) -> list[str]:
        if self.answer == "yes" or self.answer == "no":
            return [self.answer]
        elif self.answer == "unknown":
            return UNKNONW_ANSWERS
        else:
            raise ValueError("'answer' must be 'yes', 'no', or 'unknown'")

    def get_response_template(self):
        e0, e1 = self.e0.entity.name, self.e1.entity.name
        l0, l1 = self.e0.coordinate.name, self.e1.coordinate.name
        r_base, r_opposite = random.choice(self.r.base), random.choice(self.r.opposite)
        return {
            "unknown": (
                f"{REPLACE_PLACEHOLDER} if {e0} went to the {l0} {r_base} "
                f"or {r_opposite} {e1} went to the {l1}."
            ),
            "yes": f"{REPLACE_PLACEHOLDER}, {e0} went to the {l0} {r_base} {e1} went to the {l1}.",
            "no": f"{REPLACE_PLACEHOLDER}, {e0} did not go to the {l0} before {e1} went to the {l1}.",
        }


class TemporalTrackingWhenRequest(TemporalTrackingRequest):
    answer: Literal["designated_relation", "opposite_designated_relation", "unknown"]

    def get_question(self):
        question_template = "When did {e0} go to the {l0}, relative to {e1}'s going to {l1}?"
        e0, e1 = self.e0.entity.name, self.e1.entity.name
        l0, l1 = self.e0.coordinate.name, self.e1.coordinate.name
        return question_template.format(e0=e0, e1=e1, l0=l0, l1=l1)

    def get_answer(self) -> list[str]:
        if self.answer == "designated_relation":
            return ["before"]
        elif self.answer == "opposite_designated_relation":
            return ["after"]
        elif self.answer == "unknown":
            return UNKNONW_ANSWERS
        else:
            raise ValueError("'answer' must be 'designated_relation', 'opposite_designated_relation', or 'unknown'")

    def get_response_template(self):
        e0, e1 = self.e0.entity.name, self.e1.entity.name
        l0, l1 = self.e0.coordinate.name, self.e1.coordinate.name
        r_base, r_opposite = random.choice(self.r.base), random.choice(self.r.opposite)
        return {
            "unknown": (
                f"{REPLACE_PLACEHOLDER} if {e0} went to the {l0} {r_base} "
                f"or {r_opposite} {e1} went to the {l1}."
            ),
            "designated_relation": f"{e0} went to the {l0} {REPLACE_PLACEHOLDER} {e1} went to the {l1}.",
            "opposite_designated_relation": f"{e0} went to the {l0} {REPLACE_PLACEHOLDER} {e1} went to the {l1}.",
        }


class TemporalTrackingWhereRequest(TemporalTrackingRequest):
    answer: Literal["designated_location", "none"]

    def get_question(self):
        question_template = "Where did {e0} go before {e1} went to the {l1}?"
        e0, e1 = self.e0.entity.name, self.e1.entity.name
        l0, l1 = self.e0.coordinate.name, self.e1.coordinate.name
        return question_template.format(e0=e0, e1=e1, l0=l0, l1=l1)

    def get_answer(self) -> list[str]:
        if self.answer == "designated_location":
            return [self.e0.coordinate.name]
        elif self.answer == "none":
            return ["nowhere"]
        else:
            raise ValueError("'answer' must be 'designated_location' or 'none'")

    def get_response_template(self):
        e0, e1 = self.e0.entity.name, self.e1.entity.name
        _, l1 = self.e0.coordinate.name, self.e1.coordinate.name
        r_base, _ = random.choice(self.r.base), random.choice(self.r.opposite)
        return {
            "designated_location": f"{e0} went to the {REPLACE_PLACEHOLDER} {r_base} {e1} went to the {l1}.",
            "none": f"{e0} went {REPLACE_PLACEHOLDER} {r_base} {e1} went to the {l1}.",
        }


class TemporalTrackingWhoRequest(TemporalTrackingRequest):
    answer: Literal["designated_actor", "none"]

    def get_question(self):
        e0, e1 = self.e0.entity.name, self.e1.entity.name
        l0, l1 = self.e0.coordinate.name, self.e1.coordinate.name
        r_base, _ = random.choice(self.r.base), random.choice(self.r.opposite)
        question_template = "Who went to the {l0} {r_base} {e1} went to the {l1}?"
        return question_template.format(e0=e0, e1=e1, l0=l0, l1=l1, r_base=r_base)

    def get_answer(self) -> list[str]:
        if self.answer == "designated_actor":
            return [self.e0.entity.name]
        elif self.answer == "none":
            return ACTORS_NONE_ANSWERS
        else:
            raise ValueError("'answer' must be 'designated_actor' or 'none'")

    def get_response_template(self):
        _, e1 = self.e0.entity.name, self.e1.entity.name
        l0, l1 = self.e0.coordinate.name, self.e1.coordinate.name
        r_base, _ = random.choice(self.r.base), random.choice(self.r.opposite)
        return {
            "designated_actor": f"{REPLACE_PLACEHOLDER} went to the {l0} {r_base} {e1} went to the {l1}.",
            "none": f"{REPLACE_PLACEHOLDER} went to the {l0} {r_base} {e1} went to the {l1}.",
        }


class EntitiesInCoordinatesAsEvents(EntitiesInCoordinates):
    relation: Relationship

    @property
    def as_tuple(self):
        return (
            self.coordinates,
            self.entities,
        )


class TemporalTracking(OrderBaseGenerator):
    model: EntitiesInCoordinatesAsEvents
    events_qty: int
    # graphs: Optional[list[TemporalTrackingGraph]] = None
    topic: TemporalTrackingRequest
    events: Optional[list[TemporalTrackingEvent]] = None

    @model_validator(mode="before")
    def _validate_edge_qty(cls, values):
        """Removed from the parent class to avoid conflicts with the events_qty parameter."""
        return values

    def load_ontology_from_topic(self) -> Callable:
        """
        Load the appropriate ontology generator function based on the topic type.
        
        Returns:
            Callable: The function that generates the appropriate temporal tracking scenario.
        """
        # Define the mapping between answer types and loader functions
        loader_mapping: dict[type[TemporalTrackingRequest], Callable] = {
            TemporalTrackingPolarRequest: self._temporal_tracking_polar,
            TemporalTrackingWhenRequest: self._temporal_tracking_when,
            TemporalTrackingWhereRequest: self._temporal_tracking_where,
            TemporalTrackingWhoRequest: self._temporal_tracking_who,
        }
        # Get the type of the answer
        topic_type = type(self.topic)

        return loader_mapping[topic_type]

    def _create_aux(self):
        """Create auxiliary data structures for the temporal tracking generator."""
        self.shape = (self.events_qty, self.events_qty)
        self.shape_str = self.topic.shape_str

    def _create_empty_graph(self) -> tuple[DOK, nx.Graph]:
        """
        Create an empty graph represented as a Sparse matrix and a DiGraph.
        
        Returns:
            tuple[DOK, nx.Graph]: A tuple containing the sparse adjacency matrix and the graph.
        """
        g_am = DOK((self.shape[1], self.shape[1]), fill_value=np.nan)
        g = nx.DiGraph()
        for i, e in enumerate(self.events):
            g_am[i, i] = 0  # diagonal is 0
            g.add_node(i, event=e)
        return g_am, g

    def _init_setup(self) -> tuple[SparseArray, nx.DiGraph]:
        """
        Setup the initial arguments of our generator, and fill the elements in the topic/scenario.
        
        Returns:
            tuple: A tuple containing the sparse adjacency matrix and the graph.
        """
        first_different = self.topic.answer == "unknown"
        self.events = self._generate_events(
            self.model.entities,
            self.model.coordinates,
            self.events_qty,
            first_different,
        )
        self.topic.e0 = self.events[0]
        self.topic.e1 = self.events[1]
        self._create_aux()
        r_am, r = self._create_empty_graph()
        return r_am, r

    def _fill_edge_events(
        self,
        events: list[TemporalTrackingEvent],
        g: nx.Graph,
        matrix: SparseArray,
        edges_qty: int,
        condition: Callable,
        condition_degree: Callable,
    ) -> tuple[SparseArray, nx.Graph]:
        g_aux = deepcopy(g)
        condition_degree_aux = lambda g: True
        while condition_degree_aux(g_aux):
            g_loop = deepcopy(g)
            matrix_aux = deepcopy(matrix)
            condition_degree_aux = condition_degree
            # Custom comparator, sort by first element of event
            # Events should be sortable in lexicographical order
            # SortedEvents = sort(enumerate(Events), key=lambda x: x[1][0])
            sorted_events = sorted(events, key=lambda event: (event.entity.name, event.coordinate.name))
            # list of lists, grouped by first element of event
            # GroupedEvents = [list(group) for key, group in groupby(SortedEvents,
            #                  key=lambda x: x[1][0])]
            grouped_events = [list(group) for key, group in groupby(sorted_events, key=lambda x: x.entity.name)]
            while_condition = lambda x, m: True
            matrix_aux_inner = deepcopy(matrix_aux)
            while while_condition(events, matrix_aux_inner):
                while_condition = condition
                self.logger.debug(
                    "Grouped events: ",
                    grouped_events=[
                        f"{group[0].entity.name}--" + "-".join([f"({e.coordinate.name}-{e.index})" for e in group])
                        for group in grouped_events
                    ],
                )
                matrix_aux_inner = deepcopy(matrix_aux)
                g_aux2 = deepcopy(g_loop)
                for group in grouped_events:
                    # Shuffle the events within the group to create a random temporal order
                    random.shuffle(group)
                    for i in range(len(group) - 1):
                        event1 = group[i]
                        event2 = group[i + 1]
                        idx1, idx2 = event1.index, event2.index
                        # Check if a relationship between these two events is already defined
                        if np.isnan(matrix_aux_inner[idx1, idx2]):
                            # Establish a temporal order: event1 happened before event2
                            matrix_aux_inner[idx1, idx2] = 1
                            matrix_aux_inner[idx2, idx1] = 0
                            g_aux2.add_edge(idx1, idx2)
                matrix_aux_inner, g_filled = self._fill_edges(matrix_aux_inner, g_aux2, edges_qty)
                self.logger.debug("Re-trying while_condition")
            matrix_aux = matrix_aux_inner
            self.logger.debug("Re-trying condition_degree")

            try:
                # _, g_filled = self._fill_edges(matrix_aux, g_aux2, edges_qty)
                matrix_aux, g_aux = self._transitive_reduction(g_filled)
            except Exception as e:
                self.logger.warning(
                    "_fill_edge_events failed, probably couldn't reach exactly edges_qty for the graph, trying again.",
                    error=e,
                )
                g_aux = deepcopy(g)  # Reset g_aux to ensure condition_degree is re-evaluated correctly
                continue

        return matrix_aux, g_aux

    def _generate_events(
        self,
        entities: list[Entity],
        coordinates: list[Coordinate],
        events_qtty: int,
        first_different: bool,
    ) -> list[TemporalTrackingEvent]:
        """
        Generate a list of temporal tracking events.
        
        Args:
            entities: List of available entities.
            coordinates: List of available coordinates.
            events_qtty: Number of events to generate.
            first_different: If True, ensures the first two events have different entities.
            
        Returns:
            List of TemporalTrackingEvent instances.
        """
        assert len(entities) * len(coordinates) >= events_qtty, "Not enough entities and coordinates to generate events"
        while True:
            sampled_events = random.sample([(e, c) for e in entities for c in coordinates], events_qtty)
            if first_different:
                if sampled_events[0][0] != sampled_events[1][0]:
                    break
            else:
                break
        return [TemporalTrackingEvent(entity=e, coordinate=c, index=i) for i, (e, c) in enumerate(sampled_events)]

    def condition_degree(self, g):
        """
        Check if the graph satisfies the degree conditions (# of hops) based on the topic answer.
        
        Args:
            g: NetworkX directed graph representing event relationships.
            
        Returns:
            bool: True if the condition needs to be re-evaluated, False otherwise.
        """
        if self.topic.answer in ["yes", "designated_relation", "designated_location", "designated_actor"]:
            if nx.has_path(g, source=0, target=1):
                return nx.shortest_path_length(g, source=0, target=1) < 2  # self.n_hops
            else:
                return True
        elif self.topic.answer in ["no", "opposite_designated_relation"]:
            if nx.has_path(g, source=1, target=0):
                return nx.shortest_path_length(g, source=1, target=0) < 2  # self.n_hops
        elif self.topic.answer in ["unknown", "none"]:
            return nx.has_path(g, source=0, target=1)

    def _temporal_tracking_polar(self):
        graphs = []
        r_am, r = self._init_setup()

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
        else:
            raise ValueError("'answer' must be 'yes', 'no', or 'unknown'")

        condition = lambda x, m: False
        r_am, r = self._fill_edge_events(
            self.events, r, r_am, edges_qty=self.edge_qty, condition=condition, condition_degree=self.condition_degree
        )
        graphs.append(
            TemporalTrackingGraph(
                am=r_am,
                g=r,
                name="relative_event",
                index=0,
                events=self.events,
            )
        )
        return graphs

    def _temporal_tracking_when(self):
        graphs = []
        r_am, r = self._init_setup()

        if self.topic.answer == "designated_relation":
            r_am[0, 1] = 1
            r_am[1, 0] = 0
            r.add_edge(0, 1)
        elif self.topic.answer == "opposite_designated_relation":
            r_am[0, 1] = 0
            r_am[1, 0] = 1
            r.add_edge(1, 0)
        elif self.topic.answer == "unknown":
            r_am[0, 1] = 0
            r_am[1, 0] = 0
        else:
            raise ValueError("'answer' must be 'designated_relation', 'opposite_designated_relation', or 'unknown'")

        condition = lambda x, m: False  # No condition for now
        r_am, r = self._fill_edge_events(
            self.events, r, r_am, edges_qty=self.edge_qty, condition=condition, condition_degree=self.condition_degree
        )
        graphs.append(
            TemporalTrackingGraph(
                am=r_am,
                g=r,
                name="relative_event",
                index=0,
                events=self.events,
            )
        )
        return graphs

    def _temporal_tracking_where(self):
        graphs = []
        r_am, r = self._init_setup()

        if self.topic.answer == "designated_location":
            r_am[0, 1] = 1
            r_am[1, 0] = 0
            r.add_edge(0, 1)
            # condition = any(Events[2:], lambda x :
            # x[0] == Events[0][0] and relation[Events[1].index][x.index] != 1)
            condition = lambda x, m: any(x_i.entity == x[0].entity and m[x[1].index, x_i.index] != 1 for x_i in x[2:])
        elif self.topic.answer == "none":
            # condition = any(Events, lambda x :
            # x[0] == Events[0][0] and relation[Events[1].index][x.index] != 1)
            condition = lambda x, m: any(
                x_i.entity == x[0].entity and x_i != x[1] and m[x[1].index, x_i.index] != 1 for x_i in x
            )

        else:
            raise ValueError("'answer' must be 'designated_location' or 'none'")

        r_am, r = self._fill_edge_events(
            self.events, r, r_am, edges_qty=self.edge_qty, condition=condition, condition_degree=self.condition_degree
        )
        graphs.append(
            TemporalTrackingGraph(
                am=r_am,
                g=r,
                name="relative_event",
                index=0,
                events=self.events,
            )
        )
        return graphs

    def _temporal_tracking_who(self):
        graphs = []
        r_am, r = self._init_setup()
        if self.topic.answer == "designated_actor":
            r_am[0, 1] = 1
            r_am[1, 0] = 0
            r.add_edge(0, 1)
            # x = List[Events], M=Matrix of relations
            # condition = any(Events[2:], lambda x : x[1] == Events[0][1] and relation[Events[1].index][x.index] != 1)
            condition = lambda x, m: any(
                x_i.coordinate == x[0].coordinate and m[x[1].index, x_i.index] != 1 for x_i in x[2:]
            )
        elif self.topic.answer == "none":
            # condition = any(Events, lambda x : x[1] == Events[0][1] and relation[Events[1].index][x.index] != 1)
            condition = lambda x, m: any(
                x_i.coordinate == x[0].coordinate and x_i != x[1] and m[x[1].index, x_i.index] != 1 for x_i in x
            )
        else:
            raise ValueError("'answer' must be 'designated_actor' or 'none'")

        r_am, r = self._fill_edge_events(
            self.events, r, r_am, edges_qty=self.edge_qty, condition=condition, condition_degree=self.condition_degree
        )
        graphs.append(
            TemporalTrackingGraph(
                am=r_am,
                g=r,
                name="relative_event",
                index=0,
                events=self.events,
            )
        )
        return graphs

    def create_fol(self):
        """
        Create First-Order Logic (FOL) representation of the temporal tracking scenario.
        
        This method generates:
        1. World enumeration - lists all entities and coordinates
        2. Story - describes temporal relationships between events
        3. Question and answer based on the topic
        """
        def enumerate_model(element: list, shape_type: str) -> list[list]:
            enumeration = []
            for e in element:
                enumeration.append(Exists(thing=e, shape_str=shape_type))
            return enumeration

        def describe_relation(graph, shape_str):
            graph_sentences = []
            for i, j in graph.am.data:
                # check first am
                edge = graph.am[i, j]
                if edge == 1:
                    # then verify if the edge exists in the graph
                    assert graph.g.has_edge(i, j), "edge {}-{} does not exist in the graph".format(i, j)
                    graph_sentences.append(
                        IsTemporalRelated(
                            event0=graph.events[i],
                            event1=graph.events[j],
                            relation=self.model.relation,
                            shape_str=shape_str,
                        )
                    )
            return graph_sentences

        world_enumerate = []
        story = []
        # World enumeration
        for t, dim_str in zip(self.model.as_tuple, self.shape_str):
            world_enumerate.extend(enumerate_model(t, dim_str))
        random.shuffle(world_enumerate)

        # Story
        for graph in self.graphs:
            story.extend(describe_relation(graph, self.shape_str))
        random.shuffle(story)

        self.story = Story(
            world_enumerate=world_enumerate,
            describe_len=0,
            story=story,
            question=self.topic.get_question(),
            answer=self.topic.get_answer(),
            response_templates=self.topic.get_response_template(),
        )
        # FOL
        self.fol = world_enumerate + story

    def get_json(self):
        """
        Generate a JSON representation of the temporal tracking scenario.
        
        Returns:
            dict: A dictionary containing the story, question, answer, options,
                  and contextualized responses.
        """
        from typing import get_type_hints

        json = self.story.create_json()
        options = list(get_type_hints(self.topic)["answer"].__args__)
        contextualized_options = dict()
        if isinstance(self.topic, TemporalTrackingPolarRequest):
            contextualized_options["yes"] = ["yes"]
            contextualized_options["no"] = ["no"]
            contextualized_options["unknown"] = [UNKNONW_ANSWERS[0]]
        elif isinstance(self.topic, TemporalTrackingWhenRequest):
            options.remove("designated_relation")
            options.remove("opposite_designated_relation")
            designated_r_opt, op_designated_r_opt = "before", "after"
            contextualized_options["designated_relation"] = [designated_r_opt]
            contextualized_options["opposite_designated_relation"] = [op_designated_r_opt]
            contextualized_options["unknown"] = [UNKNONW_ANSWERS[0]]
            # add to options
            options.extend([designated_r_opt, op_designated_r_opt])
        elif isinstance(self.topic, TemporalTrackingWhereRequest):
            options = []
            # list coords
            designated_location_opt = [e.name for e in self.model.coordinates]
            contextualized_options["designated_location"] = designated_location_opt
            contextualized_options["none"] = ["nowhere"]
            # add to options
            options.extend(designated_location_opt)
            options.extend(["nowhere"])
        elif isinstance(self.topic, TemporalTrackingWhoRequest):
            # clean options. Neither 'designated_actor' nor 'none' are types of answers
            options = []
            designated_actor_options = [e.name for e in self.model.entities]
            contextualized_options["designated_actor"] = designated_actor_options
            # take one none option
            none_option = random.choice(ACTORS_NONE_ANSWERS)
            contextualized_options["none"] = [none_option]
            # add to options
            options.extend(designated_actor_options)
            options.extend([none_option])

        random.shuffle(options)
        json["options"] = options

        # Add contextualized responses
        json["contextualized_options"] = list()
        for key in contextualized_options:
            random.shuffle(contextualized_options[key])
            for element in contextualized_options[key]:
                json["contextualized_options"].append(
                    self.story.response_templates[key].replace(REPLACE_PLACEHOLDER, element)
                )

        json["contextualized_answer"] = list()
        for element in self.story.answer:
            # This logic might need refinement based on answer types.
            if self.topic.answer not in [
                "yes",
                "no",
                "unknown",
                "designated_relation",
                "opposite_designated_relation",
                "designated_location",
                "designated_actor",
                "none",
            ]:
                json["contextualized_answer"].append(
                    self.story.response_templates["pass"].replace(REPLACE_PLACEHOLDER, element)
                )
            else:
                json["contextualized_answer"].append(
                    self.story.response_templates[self.topic.answer].replace(REPLACE_PLACEHOLDER, element)
                )

        if self.name and DELIM in self.name:
            parts = self.name.split(DELIM)
            if len(parts) == 3:
                json["leaf"] = parts[0]
                json["leaf_label"] = parts[1]
                json["leaf_index"] = parts[2]
            else:
                raise ValueError(f"self.name does not contain exactly three parts separated by {DELIM}")
        else:
            raise ValueError(f"self.name is either None or does not contain the delimiter {DELIM}")

        return json
