import random
from typing import Callable, get_type_hints

import networkx as nx
import numpy as np

from sparse import SparseArray

from babisteps.basemodels.generators import (
    DELIM,
    OrderBaseGenerator,
    OrderRequest,
    OrderRequestPolar,
    OrderRequestHow,
)
from babisteps.basemodels.nodes import ImmediateGraph


class GeneralOrder(OrderBaseGenerator):
    topic: OrderRequest

    def load_ontology_from_topic(self) -> Callable:
        # Define the mapping between answer types and loader functions
        loader_mapping: dict[type[OrderRequest], Callable] = {
            OrderRequestPolar: self._order_polar,
            OrderRequestHow: self._order_how,
        }
        # Get the type of the answer
        topic_type = type(self.topic)

        return loader_mapping[topic_type]

    def _get_rnd_hops(self) -> np.array:
        # TODO: Add description !
        # currently this functions limit the number of
        # hops checking if the number of edges that will appear
        # after the initial transitive closure is lower than
        # self.edge_qty-1
        def _check_hops(hops) -> bool:
            """
            Check if the number of hops is valid w.r.t the number of edges
            """
            # # of transitive closure edges given the number of hops
            sum_tc = sum(range(hops+1))
            return sum_tc < self.edge_qty-1

        # range of entities (staring from to avoid e0 and e1)
        range_e = np.arange(2, len(self.model.entities))
        # get valids random hops
        flag = True
        while flag:
            hop_range = range(1, len(self.model.entities)-1)
            n_hops = random.choice(hop_range)
            if _check_hops(n_hops):
                flag = False
        # get random entities that confor the hops
        array_rnd_e = np.random.choice(range_e, max(1, n_hops-1), replace=False)
        # add e0 and e1 to the array
        array_rnd_e = np.insert(array_rnd_e, 0, 0)
        array_rnd_e = np.append(array_rnd_e, 1)
        return array_rnd_e

    def _transitive_reduction(self,
        g: nx.DiGraph
    )->tuple[SparseArray,nx.DiGraph]:
        """
        Perform transitive reduction on the given adjacency matrix and graph.
        """
        TR = nx.transitive_reduction(g)
        TR.add_nodes_from(g.nodes(data=True))
        TR.add_edges_from((u, v, g.edges[u, v]) for u, v in TR.edges)
        # Genereate an empty adjacency matrix with graph after
        # transitive reduction
        am , _= self._create_empty_graph()
        # Fill the adjacency matrix with the edges from the transitive reduction
        for u, v in TR.edges():
            am[u, v] = 1
            am[v, u] = 0
        return am, TR    
        
    def _order_polar(self):
        graphs = []
        e0, e1, r_am, r = self._init_setup()
        if self.topic.answer == "yes":
            array_rnd_e = self._get_rnd_hops()
            for i_rnd_e in range(len(array_rnd_e)-1):
                n_i = array_rnd_e[i_rnd_e]
                n_f = array_rnd_e[i_rnd_e + 1]
                r_am[n_i, n_f], r_am[n_f, n_i] = 1, 0
                r.add_edge(n_i, n_f)
        elif self.topic.answer == "no":
            array_rnd_e = self._get_rnd_hops()
            for i_rnd_e in range(len(array_rnd_e)-1):
                n_i = array_rnd_e[i_rnd_e]
                n_f = array_rnd_e[i_rnd_e + 1]
                r_am[n_i, n_f], r_am[n_f, n_i] = 0, 1
                r.add_edge(n_f, n_i)
        elif self.topic.answer == "unknown":
            r_am[0, 1] = 0
            r_am[1, 0] = 0

        self.logger.debug(
            "Creating OrderPolar",
            answer=self.topic.answer,
            e0=e0.name,
            e1=e1.name,
            relation=self.topic.r.name,
        )
        r_am, r = self._fill_edges(r_am, r, self.edge_qty)
        # Get updated am and graph performing a transitive reduction.
        r_am, r = self._transitive_reduction(r)

        graphs.append(
            ImmediateGraph(am=r_am,
                           g=r,
                           name=self.model.relations[0].name,
                           index=0))
        if len(self.model.relations) > 1:
            for i, rlt in enumerate(self.model.relations[1:], start=1):
                g_am, g = self._create_empty_graph()
                g_am, g = self._fill_edges(g_am, g, self.edge_qty)
                g_am, g  = self._transitive_reduction(g)
                graphs.append(
                    ImmediateGraph(am=g_am, g=g, name=rlt.name, index=i))
        return graphs

    def _order_how(self):
        graphs = []
        e0, e1, r_am, r = self._init_setup()
        if self.topic.answer == "designated_relation":
            array_rnd_e = self._get_rnd_hops()
            for i_rnd_e in range(len(array_rnd_e)-1):
                n_i = array_rnd_e[i_rnd_e]
                n_f = array_rnd_e[i_rnd_e + 1]
                r_am[n_i, n_f], r_am[n_f, n_i] = 1, 0
                r.add_edge(n_i, n_f)
        elif self.topic.answer == "unknown":
            r_am[0, 1] = 0
            r_am[1, 0] = 0
        else:
            raise ValueError(
                "Invalid answer should be 'designated_relation' or 'unknown'")
        self.logger.debug(
            "Creating OrderHow",
            answer=self.topic.answer,
            e0=e0.name,
            e1=e1.name,
            relation=self.topic.r.name,
        )
        r_am, r = self._fill_edges(r_am, r, self.edge_qty)
        r_am, r = self._transitive_reduction(r)
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
                g_am, g  = self._transitive_reduction(g)
                graphs.append(
                    ImmediateGraph(am=g_am, g=g, name=rlt.name, index=i))
        return graphs

    def get_json(self):
        json = self.story.create_json()
        options = list(get_type_hints(self.topic)["answer"].__args__)
        if isinstance(self.topic, OrderRequestPolar):
            pass
        elif isinstance(self.topic, OrderRequestHow):
            options.remove("designated_relation")
            o = self.topic.get_options(self.model.relations)
            options.extend(o)


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
                    f"self.name does not contain exactly three parts "
                    f"separated by {DELIM}"
                )
        else:
            raise ValueError(
                f"self.name is either None or does not contain the delimiter {DELIM}"
            )

        return json
