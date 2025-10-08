import random
from itertools import combinations
from abc import abstractmethod
from typing import Any, Callable, Literal, Optional, Union, get_type_hints

import networkx as nx
import numpy as np
from pydantic import Field, BaseModel
from sparse import DOK, SparseArray

from babisteps.basemodels.generators import (
    DELIM,
    OrderBaseGenerator,
    UNKNONW_ANSWERS,
    REPLACE_PLACEHOLDER,
)

from babisteps.basemodels.listing import ANSWER_OPTION_QTY

from babisteps.basemodels.nodes import Entity, PathFindingGraph, Relationship

MAX_ATTEMPTS_MULTIPLIER = 10

class PathFindingRequest(BaseModel):
    answer: Any
    path: Any = None

    @abstractmethod
    def get_question(self):
        """Abstract method to generate the strings' question"""
        pass

    @abstractmethod
    def get_answer(self):
        """Abstract method to generate the answer"""
        pass

    @abstractmethod
    def get_reponse_tempalte(self):
        """Abstract method to generate the answer context template"""
        pass

class PathFindingRequestWhich(PathFindingRequest):
    answer: Union[int, Literal["unknown"]]
    path: Optional[list[Entity]] = None
    path_graph: Optional[PathFindingGraph] = None
    relation_type: Literal["absolute_position"]
    shape_str: tuple[Literal[("locations",)]] = ("locations",)
    # NOTE: `relations` nomeclature is as follow:
    # [r_{i}, (l_{j}, l_{j+1}), r_{i+1}, (l_{j+1}, l_{j+2})...]
    # where r_{i} is the index of the relation
    # and l_{j}, l_{j+1} are the indexes of the consecutive relations
    relations: Optional[list[list]] = None


    def get_question(self):
        question_template = "Which path goes from the {p} to the {q}?"
        p = self.path[0].name
        q = self.path[-1].name
        return question_template.format(p=p, q=q)
        
    def get_answer(self) -> list[str]:
        if self.answer == "unknown":
            return UNKNONW_ANSWERS
        elif isinstance(self.answer, int):
            return [location.name for location in self.path]

    def get_reponse_tempalte(self):
        return {
            "unknown": 
            f"{REPLACE_PLACEHOLDER} how to get from {self.path[0].name} to {self.path[-1].name}.",
            "path": 
            f"The path that goes from {self.path[0].name} to {self.path[-1].name} is: {REPLACE_PLACEHOLDER}."
        }

class PathFinding(OrderBaseGenerator):
    graphs: Optional[list[PathFindingGraph]] = None
    topic: PathFindingRequest
    path_length: int = Field(ge=3)

    def load_ontology_from_topic(self) -> Callable:
        # Define the mapping between answer types and loader functions
        loader_mapping: dict[type[PathFindingRequest], Callable] = {
            PathFindingRequestWhich: self._path_finding_which,
        }
        # Get the type of the answer
        topic_type = type(self.topic)

        return loader_mapping[topic_type]

    def _create_empty_graph(self) -> tuple[DOK, nx.Graph]:
        """
        Create an empty graph represented as a Sparse matrix and a DiGraph
        """
        g_am = DOK((self.shape[1], self.shape[1]), fill_value=np.nan)
        g = nx.Graph()
        for i, e in enumerate(self.model.entities):
            g_am[i, i] = 0  # diagonal is 0
            g.add_node(i, entity=e)
        return g_am, g

    def _init_setup(self) -> tuple[Entity, Entity, SparseArray, nx.DiGraph]:
        """
        Setup the initial args of our generator, and
        fill the elements in the topic/scenario.
        """
        self._create_aux()
        self.topic.path = self.model.entities[:self.path_length]
        r_am, r = self._create_empty_graph()

        return r_am, r

    def get_random_combinations(self, n):
        """
        Returns n random combinations (subsets) of the path list.
        The empty set and the full set are included in the possible combinations.
        Order doesn't matter.
        
        Args:
            string_list: List of strings
            n: Number of random combinations to return

        Returns:
            List of n random combinations (as lists)
        """
        string_list = [e.name for e in self.model.entities]
        e_i= self.topic.path[0].name
        e_f= self.topic.path[-1].name
        string_list.remove(e_i)
        string_list.remove(e_f)
        # Calculate total possible combinations: 2^len(string_list)
        total_possible = 2**len(string_list)

        # If n >= total possible combinations, generate all combinations
        if n >= total_possible:
            result = []  # Start with empty set
            for r in range(1, len(string_list) + 1):
                for comb in combinations(string_list, r):
                    result.append(list(comb))
            return result

        # Generate n random unique combinations
        result = set()

        # Ensure we have n unique combinations
        while len(result) < n:
            # Choose a random size for this combination
            size = random.randint(2, len(string_list))
            comb = tuple(sorted(random.sample(string_list, size)))
            result.add(comb)

        # Convert set of tuples back to list of lists
        return [list(comb) for comb in result]    

    def _extend_preserving_unique_path(self, connections_am, connections_g, edge_qty) -> tuple[DOK, nx.Graph]:
        """
        REQUIRES: connections_am.width = connections_am.height
        Add edges (randomly) to connections_g one at a time checking that with each addition there is one and only
        one path from start to end. If that is meet, add the edge to connections_am as well.
        Repeat until edge_qty edges have been added. Creat
        """
        attempts = 0
        max_attempts = edge_qty * MAX_ATTEMPTS_MULTIPLIER  # Arbitrary limit to prevent infinite loops
        current_edge_qty_am = np.sum((connections_am > 0).to_coo()) // 2  # Each edge is counted twice in an undirected graph
        current_edge_qty = connections_g.number_of_edges()
        assert current_edge_qty_am == current_edge_qty, (
            f"Mismatch between adjacency matrix edges and graph edges: "
            f"{current_edge_qty_am} vs {current_edge_qty}"
        )
        while current_edge_qty < edge_qty and attempts < max_attempts:
            # Randomly select two distinct nodes
            n1, n2 = random.sample(range(connections_am.shape[0]), 2)
            # if both n1 and n2 are < path_length:, continue
            # locations n1 and n2 are already connected in the path, so skip them
            if n1 < self.path_length and n2 < self.path_length:
                continue
            # Only consider adding the edge if it doesn't already exist
            if connections_am[n1, n2] != 1:
                self.logger.debug("Connections between %s and %s is currently %s", n1, n2, connections_am[n1, n2])
                # Add the edge temporarily
                connections_g.add_edge(n1, n2)
                # Check if there is exactly one path from start to end
                try:
                    paths = list(nx.all_simple_paths(connections_g, source=0, target=self.path_length-1))
                    if len(paths) == 1:
                        # If there is exactly one path, make the addition permanent
                        connections_am[n1, n2] = 1
                        connections_am[n2, n1] = 1
                        current_edge_qty += 1
                        self.logger.debug("Successfully added edge between %s and %s (attempt %d)", n1, n2, attempts)
                    else:
                        # Otherwise, remove the edge
                        connections_g.remove_edge(n1, n2)
                except nx.NetworkXNoPath:
                    # If no path exists, remove the edge
                    connections_g.remove_edge(n1, n2)
            attempts += 1

        if attempts == max_attempts:
            self.logger.error("Max attempts reached while extending the graph.")
            raise RuntimeError("Failed to extend graph while preserving unique path.")
        return connections_am, connections_g

    def _path_finding_which(self):
        graphs = []
        path_len = self.path_length
        c_am, c = self._init_setup()

        # NOTE: c_am and c refer to the connections adjacency matrix/graph
        # Link every consecutive pair in the path_len
        for i in range(path_len - 1):
            c_am[i, i + 1] = 1
            c_am[i + 1, i] = 1
            # add the edge to the graph
            c.add_edge(i, i + 1)

        # NOTE: Compare to other approach, where c_am and c refered to relations matrix/grahps
        # this time, there refers to the connections matrix/graph
        # Due to this, now its neecesary to create the relations matrix/graph for each relation.
        # Iterate over each relation available
        relation_elements = []
        relation_index_result = []
        for i, relation in enumerate(self.model.relations):
            r_am_i, r_i = self._create_empty_graph()
            # NOTE: relations elements type (int, Relationship, DOK, nx.Graph)
            relation_elements.append((i, relation, r_am_i, r_i))
        if self.topic.answer == "unknown":
            # Decide how many links to disconnect
            # number_of_subtractions = choose([1, path_length - 1])
            number_of_subtractions = random.randint(1, path_len - 1)
            # Decide links from what locations to disconnect (sampling without replacement)
            subtractions = random.sample(range(0, path_len), number_of_subtractions)
        else:
            number_of_subtractions = 0
            subtractions = []

        try:
            # Add connections while preserving a unique path
            c_am, c = self._extend_preserving_unique_path(c_am, c, self.edge_qty + number_of_subtractions)
        except Exception as e:
            self.logger.error("Error while extending graph: %s", e)
            raise e

        # When corresponds, disconnect the previously selected edges
        for s in subtractions:
            c_am[s, s + 1] = 0
            c_am[s + 1, s] = 0
            c.remove_edge(s, s + 1)

        # Iterate over each the low-triangular part of the connections matrix
        for i in range(len(self.model.entities)):
            for j in range(i):
                # if Connections[i][j] == 1:
                if c_am[i, j] == 1:

                    # choose relation in Relation:
                    r_index, relation_i, r_am_i, r_i = random.choice(relation_elements)
                    if i == j + 1 and self.topic.answer != "unknown":
                        relation_index_result.append([r_index, (i, j)])
                    if random.random() > 0.5:
                        r_am_i[i, j] = 1
                        r_am_i[j, i] = 0
                        r_i.add_edge(i, j)
                    else:
                        r_am_i[j, i] = 1
                        r_am_i[i, j] = 0
                        r_i.add_edge(j, i)

                    # for other_relation in Relation \ relation:
                    for _, other_relation, other_am, other_r in relation_elements:
                        if other_relation != relation_i:
                            other_am[i, j] = 0
                            other_am[j, i] = 0
                            # NOTE: In networkx, removing a non-existing edge does raise an error
                            # so check first if they exist
                            if other_r.has_edge(i, j):
                                other_r.remove_edge(i, j)
                            if other_r.has_edge(j, i):
                                other_r.remove_edge(j, i)

        for i, (r_index, relation_i, r_am_i, r_i) in enumerate(relation_elements):
            graphs.append(PathFindingGraph(am=r_am_i, g=r_i, name=relation_i.name, index=i))
    
        self.topic.path_graph = graphs
        self.topic.relations = relation_index_result

        return graphs

    def get_json(self):
        json = self.story.create_json()
        options = list()
        contextualized_options = dict()
        if isinstance(self.topic, PathFindingRequest):
            unknown_option = random.choice(UNKNONW_ANSWERS)
            contextualized_options["unknown"] = [unknown_option]
            options.append([unknown_option])
            
        else:
            raise ValueError("Invalid topic type for PathFinding generator")

        o = self.get_random_combinations(n=ANSWER_OPTION_QTY)
        options.extend(o)

        # Derive 'anws' from 'self.topic.entities'
        if isinstance(self.topic.answer, int):
            if self.topic.path is None:
                raise ValueError(
                    "self.topic.path is None, cannot derive 'anws'")
            anws = sorted([e.name for e in self.topic.path[1:-1]])
            # if is not in the options, add it
            if anws not in options:
                options.append(anws)
   
        e_i = self.topic.path[0].name
        e_f = self.topic.path[-1].name
        # adding to all elements in options
        options = [ [e_i] + opt + [e_f] if len(opt)>=2 else opt for opt in options]

        # Shuffle to avoid bias in the answer order
        random.shuffle(options)
        json["options"] = options

        contextualized_options["path"] = list()
        for opts_i in options:
            # normal paths
            if len(opts_i) > 2:
                list_text = ", ".join(opts_i[:-1])
                list_text += f" and {opts_i[-1]}"
                contextualized_options["path"].append(list_text)
            # this should be the case unknown
            elif len(opts_i) == 1:
                continue
            else:
                # should not happen, then raise an error
                raise ValueError("Options with only two elements should not happen")

        # Add contextualized responses
        json["contextualized_options"] = list()
        for key in contextualized_options:
            random.shuffle(contextualized_options[key])
            for element in contextualized_options[key]:
                json["contextualized_options"].append(
                    self.story.response_templates[key].replace(
                        REPLACE_PLACEHOLDER, element))
        json["contextualized_answer"] = list()
        # If this is not a list of elements, it is a special case (like none or unknown)
        if not isinstance(self.topic.answer, int):
            for element in self.story.answer:
                json["contextualized_answer"].append(
                    self.story.response_templates[self.topic.answer].replace(
                        REPLACE_PLACEHOLDER, element))
        else:
            list_text = ", ".join(self.story.answer[:-1])
            list_text += f" and {self.story.answer[-1]}"
            json["contextualized_answer"].append(
                self.story.response_templates["path"].replace(
                    REPLACE_PLACEHOLDER, list_text))

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
