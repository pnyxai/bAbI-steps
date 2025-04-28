import random
from typing import Callable, get_type_hints

from babisteps.basemodels.generators import (DELIM, OrderBaseGenerator,
                                             OrderRequest, OrderRequestHow,
                                             OrderRequestPolar,
                                             OrderRequestWhat)
from babisteps.basemodels.nodes import ImmediateGraph


class ImmediateOrder(OrderBaseGenerator):
    topic: OrderRequest

    def load_ontology_from_topic(self) -> Callable:
        # Define the mapping between answer types and loader functions
        loader_mapping: dict[type[OrderRequest], Callable] = {
            OrderRequestPolar: self._immediate_order_polar,
            OrderRequestHow: self._immediate_order_how,
            OrderRequestWhat: self._immediate_order_what,
        }
        # Get the type of the answer
        topic_type = type(self.topic)

        return loader_mapping[topic_type]

    def _immediate_order_polar(self):
        graphs = []
        e0, e1, r_am, r = self._init_setup()
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
            "Creating ImmediateOrderPolar",
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
        graphs = []
        e0, e1, r_am, r = self._init_setup()
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
        self.logger.debug(
            "Creating ImmediateOrderHow",
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
                g_am[0, 1], g_am[1, 0] = 0, 0
                g_am, g = self._fill_edges(g_am, g, self.edge_qty)
                graphs.append(
                    ImmediateGraph(am=g_am, g=g, name=rlt.name, index=i))
        return graphs

    def _immediate_order_what(self):
        graphs = []
        e0, e1, r_am, r = self._init_setup()
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
        self.logger.debug(
            "Creating ImmediateOrderWhat",
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

    def get_json(self):
        json = self.story.create_json()
        options = list(get_type_hints(self.topic)["answer"].__args__)
        if isinstance(self.topic, OrderRequestPolar):
            pass
        elif isinstance(self.topic, OrderRequestHow):
            options.remove("designated_relation")
            o = self.topic.get_options(self.model.relations)
            options.extend(o)
        elif isinstance(self.topic, OrderRequestWhat):
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
                    f"self.name does not contain exactly three parts "
                    f"separated by {DELIM}")
        else:
            raise ValueError(
                f"self.name is either None or does not contain the delimiter {DELIM}"
            )

        return json
