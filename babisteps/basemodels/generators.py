import itertools
import logging
import random
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, model_validator
from sparse._dok import DOK

from babisteps.basemodels.FOL import FOL, Exists, From, FromTo, In, To
from babisteps.basemodels.nodes import (
    Coordenate,
    Entity,
    EntityInCoordenateState,
    ObjectInLocationState,
    ObjectInLocationStatePolar,
    State,
)
from babisteps.utils import logger, operators

# -------------------------
# Answer
# -------------------------


class SimpleTrackerRequest(BaseModel):
    answer: Any
    entity: Optional[Entity] = None
    coordenate: Optional[Coordenate] = None

    def get_question(self):
        pass

    def get_answer(self):
        pass


class ActorInLocationPolar(SimpleTrackerRequest):
    answer: Literal["yes", "no", "unknown"]

    def get_question(self):
        return f"Is {self.entity.name} in the {self.coordenate.name}?"

    def get_answer(self):
        return self.answer


class ActorInLocationWho(SimpleTrackerRequest):
    answer: Literal["designated_entity", "none", "unknown"]

    def get_question(self):
        return f"Who is in {self.coordenate.name}?"

    def get_answer(self):
        if self.answer == "designated_entity":
            return self.entity.name
        elif self.answer == "none":
            return "None"
        elif self.answer == "unknown":
            return "unknown"
        else:
            raise ValueError(
                "Invalid answer, should be 'designated_entity', 'none' or 'unknown'"
            )


class ActorInLocationWhere(SimpleTrackerRequest):
    answer: Literal["designated_location", "unknown"]

    def get_question(self):
        return f"Where is {self.entity.name}?"

    def get_answer(self):
        if self.answer == "designated_location":
            return self.coordenate.name
        elif self.answer == "unknown":
            return "unknown"


class ActorWithObjectPolar(SimpleTrackerRequest):
    answer: Literal["yes", "no"]

    def get_question(self):
        return f"Has {self.coordenate.name} the {self.entity.name}?"

    def get_answer(self):
        return self.answer


class ActorWithObjectWhat(SimpleTrackerRequest):
    answer: Literal["designated_object", "none"]

    def get_question(self):
        return f"What has {self.coordenate.name}?"

    def get_answer(self):
        if self.answer == "designated_object":
            return self.entity.name
        else:
            return "none"


class ActorWithObjectWho(SimpleTrackerRequest):
    answer: Literal["designated_actor", "none"]

    def get_question(self):
        return f"Who has the {self.entity.name}?"

    def get_answer(self):
        if self.answer == "designated_actor":
            return self.coordenate.name
        elif self.answer == "none":
            return "nobody"
        else:
            raise ValueError(
                "Invalid answer value, should be 'designated_actor' or 'unknown'"
            )


# -------------------------
# Model
# -------------------------


class EntitiesInCoordenates(BaseModel):
    entities: list[Entity]
    coordenates: list[Coordenate]

    @model_validator(mode="after")
    def _shuffle(self):
        random.shuffle(self.entities)
        random.shuffle(self.coordenates)
        return self

    @property
    def as_tuple(self):
        return (
            self.coordenates,
            self.entities,
        )


class Story(BaseModel):
    world_enumerate: list[FOL]
    describe_len: int
    story: list[FOL]
    question: str
    answer: str

    def create_json(self):
        dict_json = {}
        wd = ""
        for wd_i in self.world_enumerate:
            wd = wd + wd_i.to_nl() + " "
        s = ""
        for s_i in self.story:
            s = s + s_i.to_nl() + "\n"
        dict_json["world_enumerate"] = wd
        dict_json["story"] = s
        dict_json["question"] = self.question
        dict_json["answer"] = self.answer
        return dict_json

    def create_txt(self):
        txt = ""
        for wd_i in self.world_enumerate:
            txt += wd_i.to_nl() + "\n"

        txt += "\n"
        for idx, s_i in enumerate(self.story):
            if idx == self.describe_len:
                txt += "\n"
            txt += s_i.to_nl() + "\n"

        txt += "\n{}\n{}\n".format(self.question, self.answer)
        txt += "-" * 40
        txt += "\n\n"

        return txt


class SimpleTracker(BaseModel):
    model: Any
    states_qty: int
    topic: SimpleTrackerRequest
    uncertainty: Optional[Coordenate] = None
    verbosity: Union[int, str] = Field(default=logging.INFO)
    logger: Optional[Any] = None
    log_file: Optional[Path] = None
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
    shape_str: Literal[("Location", "Actor"), ("Actor", "Object")]

    @model_validator(mode="after")
    def fill_logger(self):
        if not self.logger:
            self.logger = logger.get_logger(
                "SimpleTracker", level=self.verbosity, log_file=self.log_file
            )
        return self

    @model_validator(mode="after")
    def check_shape_and_model(self):
        model_tuple = self.model.as_tuple
        if len(model_tuple) != len(self.shape_str):
            raise ValueError(
                f"Length mismatch: 'model.as_tuple()' has length {len(model_tuple)} "
                f"but 'shape_str' has length {len(self.shape_str)}."
            )
        return self

    def load_ontology_from_topic(self) -> Callable:
        # Define the mapping between answer types and loader functions
        loader_mapping: dict[type[SimpleTrackerRequest], Callable] = {
            ActorInLocationPolar: self._actor_in_location_polar,
            ActorInLocationWho: self._actor_in_location_who,
            ActorInLocationWhere: self._actor_in_location_where,
            ActorWithObjectPolar: self._actor_with_object_polar,
            ActorWithObjectWhat: self._actor_with_object_what,
            ActorWithObjectWho: self._actor_with_object_who,
        }
        uncertainty_mapping: dict[type[SimpleTrackerRequest], Coordenate] = {
            ActorInLocationPolar: Coordenate(name="nowhere"),
            ActorInLocationWho: Coordenate(name="nowhere"),
            ActorInLocationWhere: Coordenate(name="nowhere"),
            ActorWithObjectPolar: None,
            ActorWithObjectWhat: None,
            ActorWithObjectWho: Coordenate(name="nobody"),
        }

        # Get the type of the answer
        answer_type = type(self.topic)

        if answer_type not in loader_mapping:
            raise ValueError(
                f"Unsupported answer type: {answer_type.__name__}. "
                f"Should be one of {[cls.__name__ for cls in loader_mapping]}"
            )
        # Set the uncertainty based on the answer type
        if uncertainty_mapping[answer_type]:
            self.uncertainty = uncertainty_mapping[answer_type]

        return loader_mapping[answer_type]

    def _actor_in_location_polar(self):
        """
        Creates an ontology based on the current state of entities and their
        coordenates.
        This method initializes and updates the states of entities (entities) in
        various coordenates
        based on the provided answer. The states are created and modified according to
        the following rules:
        - If `answer` is 1: The entity `e` is in coord `c`.
        - If `answer` is 0: The entity `e` is not in coord `c` and not in `uncertainty`.
        - If `answer` is 2: Randomly decides between two conditions:
            - The entity `e` is in `uncertainty` from the beginning.
            - The entity `e` is in coord `c` at step i, and then moved to `uncertainty`.
        """

        e = self.model.entities[0]
        c = self.model.coordenates[0]

        self.topic.entity = e
        self.topic.coordenate = c
        self.model.coordenates.append(self.uncertainty)
        self._create_aux()
        self.logger.info(
            "Creating _actor_in_location_polar",
            answer=self.topic.answer,
            e=e.name,
            c=c.name,
        )
        states = [None] * self.states_qty

        if self.topic.answer == "yes":
            i = self.states_qty - 1
            condition = lambda x: x[0, 0] == 1
            states[i] = self.initialize_state(i, condition)
            for j in list(reversed(range(i))):
                condition = lambda x: True
                states[j] = self.create_new_state(j, states[j + 1], condition)

        elif self.topic.answer == "no":
            if random.choice([0, 1]):
                # case for entity in coord different from c
                i = self.states_qty - 1
                condition = lambda x: x[0, 0] == 0 and x[-1, 0] == 0
                states[i] = self.initialize_state(i, condition)
                for j in list(reversed(range(i))):
                    condition = lambda x: True
                    states[j] = self.create_new_state(j, states[j + 1], condition)
            else:
                # case where e is in uncertinty, but previously was in c.
                i = random.randint(0, self.states_qty - 2)
                condition = lambda x: x[0, 0] == 1
                states[i] = self.initialize_state(i, condition)
                for j in list(reversed(range(i))):
                    condition = lambda x: True
                    states[j] = self.create_new_state(j, states[j + 1], condition)
                # create the states after i
                for j in range(i + 1, len(states)):
                    condition = lambda x: x[-1, 0] == 1
                    states[j] = self.create_new_state(j, states[j - 1], condition)

        elif self.topic.answer == "unknown":
            if random.choice([0, 1]):
                i = 0
                condition = lambda x: x[-1, 0] == 1
                states[i] = self.initialize_state(i, condition)
                for j in range(1, self.states_qty):
                    condition = lambda x: x[-1, 0] == 1
                    states[j] = self.create_new_state(j, states[j - 1], condition)
            else:
                i = random.randint(0, self.states_qty - 2)
                condition = lambda x: x[0, 0] == 0 and x[-1, 0] == 0
                states[i] = self.initialize_state(i, condition)
                for j in list(reversed(range(i))):
                    condition = lambda x: True
                    states[j] = self.create_new_state(j, states[j + 1], condition)
                # create the states after i
                for j in range(i + 1, len(states)):
                    condition = lambda x: x[-1, 0] == 1
                    states[j] = self.create_new_state(j, states[j - 1], condition)
        else:
            raise ValueError("Invalid answer value, should be 'yes', 'no' or 'unknown'")

        self.logger.info(
            "_actor_in_location_polar successfully created",
            answer=self.topic.answer,
            e=e.name,
            c=c.name,
        )

        return states

    def _actor_in_location_who(self):
        e = self.model.entities[0]
        c = self.model.coordenates[0]
        self.topic.entity = e
        self.topic.coordenate = c
        self.model.coordenates.append(self.uncertainty)
        self._create_aux()
        self.logger.info(
            "Creating _actor_in_location_who",
            answer=self.topic.answer,
            e=e.name,
            c=c.name,
        )
        states = [None] * self.states_qty

        if self.topic.answer == "designated_entity":
            i = self.states_qty - 1
            condition = lambda x: x[0, 0] == 1 and sum(x[0, 1:]) == 0
            states[i] = self.initialize_state(i, condition)
            for j in list(reversed(range(i))):
                condition = lambda x: True
                states[j] = self.create_new_state(j, states[j + 1], condition)

        elif self.topic.answer == "none":
            self.logger.debug("Creating _actor_in_location_who with answer none")
            i = self.states_qty - 1
            condition = lambda x: sum(x[0, :]) == 0 and sum(x[-1, :]) < self.states_qty
            states[i] = self.initialize_state(i, condition)

            EIU = states[i].get_entities_in_coodenate(self.c2idx[self.uncertainty])

            if EIU:
                self.logger.debug(
                    "Entities in uncertainty",
                    EIU=EIU,
                )
                while EIU:
                    EIU = states[i].get_entities_in_coodenate(
                        self.c2idx[self.uncertainty]
                    )
                    for j in list(reversed(range(i))):
                        ue = random.choice(self.model.entities)
                        x_ue = self.e2idx[ue]
                        if x_ue in EIU:
                            self.logger.debug(
                                "Trying to place entity from NW to coordenate c",
                                entity=x_ue,
                                coordenate=self.c2idx[c],
                                left=len(EIU) - 1,
                            )
                            condition = lambda x, x_ue=x_ue: x[0, x_ue] == 1
                            EIU.remove(x_ue)
                        else:
                            condition = lambda x, EIU=EIU: all(
                                x[-1, EIU].todense() == [1] * len(EIU)
                            )
                        states[j] = self.create_new_state(j, states[j + 1], condition)
            else:
                self.logger.debug("There were not entities in uncertainty")
                for j in list(reversed(range(i))):
                    condition = lambda x: True
                    states[j] = self.create_new_state(j, states[j + 1], condition)

        elif self.topic.answer == "unknown":
            i = self.states_qty - 1
            empty_l = lambda x: sum(x[0, :]) == 0
            some_in_UN = lambda x: sum(x[-1, :]) > 0

            condition = lambda x: empty_l(x) and some_in_UN(x)
            states[i] = self.initialize_state(i, condition)
            EIU = states[i].get_entities_in_coodenate(self.c2idx[self.uncertainty])
            for j in list(reversed(range(i))):
                ue = random.choice(self.model.entities)
                x_ue = self.e2idx[ue]
                if x_ue in EIU:
                    condition = (
                        lambda x, x_ue=x_ue: x[0, x_ue] == 0 and x[-1, x_ue] == 0
                    )
                    EIU.remove(x_ue)
                else:
                    condition = lambda x: all(x[-1, EIU].todense() == [1] * len(EIU))
                states[j] = self.create_new_state(j, states[j + 1], condition)
        else:
            raise ValueError("Invalid answer value")
        self.logger.info(
            "actor_in_location_who successfully created:",
            answer=self.topic.answer,
            e=e.name,
            c=c.name,
        )
        return states

    def _actor_in_location_where(self):
        e = self.model.entities[0]
        c = self.model.coordenates[0]
        self.topic.entity = e
        self.topic.coordenate = c
        self.model.coordenates.append(self.uncertainty)
        self._create_aux()
        self.logger.info(
            "Creating _actor_in_location_where",
            answer=self.topic.answer,
            e=e.name,
            c=c.name,
        )
        states = [None] * self.states_qty

        i = self.states_qty - 1
        if self.topic.answer == "designated_location":
            condition = lambda x: x[0, 0] == 1
        elif self.topic.answer == "unknown":
            condition = lambda x: x[-1, 0] == 1
        else:
            raise ValueError(
                "Invalid answer value, should be 'designated_location' or 'unknown'"
            )
        states[i] = self.initialize_state(i, condition)
        for j in list(reversed(range(i))):
            condition = lambda x: True
            states[j] = self.create_new_state(j, states[j + 1], condition)

        self.logger.info(
            "_actor_in_location_where successfully created:",
            answer=self.topic.answer,
            e=e.name,
            c=c.name,
        )
        return states

    def _actor_with_object_polar(self):
        e = self.model.entities[0]
        c = self.model.coordenates[0]
        self.topic.entity = e
        self.topic.coordenate = c
        self.model.coordenates.append(self.uncertainty)
        self._create_aux()
        states = [None] * self.states_qty

        self.logger.info(
            "Creating _actor_with_object_polar",
            answer=self.topic.answer,
            e=e.name,
            c=c.name,
        )

        i = self.states_qty - 1
        if self.topic.answer == "yes":
            condition = lambda x: x[0, 0] == 1
        elif self.topic.answer == "no":
            condition = lambda x: x[0, 0] == 0 and x[-1, 0] == 0
        else:
            raise ValueError("Invalid answer value, should be 1 (YES) or 0 (NO)")

        states[i] = self.initialize_state(i, condition)
        for j in list(reversed(range(i))):
            condition = lambda x: True
            states[j] = self.create_new_state(j, states[j + 1], condition)

        self.logger.info(
            "_actor_with_object_polar successfully created",
            answer=self.topic.answer,
            e=e.name,
            c=c.name,
        )

        return states

    def _actor_with_object_what(self):
        e = self.model.entities[0]
        c = self.model.coordenates[0]
        self.topic.entity = e
        self.topic.coordenate = c
        self._create_aux()
        states = [None] * self.states_qty

        i = self.states_qty - 1
        if self.topic.answer == "designated_object":
            condition = lambda x: x[0, 0] == 1 and sum(x[0, 1:]) == 0
        elif self.topic.answer == "none":
            condition = lambda x: sum(x[0, :]) == 0
        else:
            raise ValueError(
                "Invalid answer value, should be 'designated_object' or 'none'"
            )

        states[i] = self.initialize_state(i, condition)
        for j in list(reversed(range(i))):
            condition = lambda x: True
            states[j] = self.create_new_state(j, states[j + 1], condition)

        self.logger.info(
            "_actor_with_object_what successfully created",
            answer=self.topic.answer,
            e=e.name,
            c=c.name,
        )
        return states

    def _actor_with_object_who(self):
        e = self.model.entities[0]
        c = self.model.coordenates[0]
        self.model.coordenates.append(self.uncertainty)
        self.topic.entity = e
        self.topic.coordenate = c
        self._create_aux()
        states = [None] * self.states_qty

        i = self.states_qty - 1
        if self.topic.answer == "designated_actor":
            condition = lambda x: x[0, 0] == 1 and sum(x[1:, 0]) == 0
        elif self.topic.answer == "none":
            condition = lambda x: x[-1, 0] == 1
        else:
            raise ValueError(
                "Invalid answer value, should be 'designated_actor' or 'none'"
            )

        states[i] = self.initialize_state(i, condition)
        for j in list(reversed(range(i))):
            condition = lambda x: True
            states[j] = self.create_new_state(j, states[j + 1], condition)

        self.logger.info(
            "_actor_with_object_who successfully created",
            answer=self.topic.answer,
            e=e.name,
            c=c.name,
        )

        return states

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
        new_state = EntityInCoordenateState(
            am=new_am, index=j, verbosity=self.verbosity, log_file=self.log_file
        )
        return new_state

    def initialize_state(self, i: int, condition: Callable) -> EntityInCoordenateState:
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

        self.logger.debug("State initialized", state=s, answer=self.topic.answer, i=i)
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
        coordenates = np.random.choice(self.shape[0], self.shape[1], replace=True)
        sparse_matrix = DOK(shape=self.shape, dtype=int, fill_value=0)
        entity_coord_pairs = list(zip(coordenates, entities))
        for x, y in entity_coord_pairs:
            sparse_matrix[x, y] = 1
        s = EntityInCoordenateState(
            am=sparse_matrix, index=i, verbosity=self.verbosity, log_file=self.log_file
        )
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
                pair = diff.data[j : j + 2]
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

    # The following could be another way to obtain the deltas in case transition for
    # higher dimentions do not came in sorted pairs.
    # DO NOT DELETE.
    # def create_transition(self):
    #     for e in ends:
    #         end = diff.coords.T[e]
    #         zeros_per_column = diff.coords.T[origins] - diff.coords.T[e]
    #         zeros_per_column = np.sum(zeros_per_column == 0, axis=0)
    #         i = np.argmax(zeros_per_column)
    #         o = diff.coords.T[o]
    #         d = np.array([e,o])
    # TODO
    def create_fol(self):
        def enumerate_model(
            element: Union[list[Entity], list[Coordenate]], shape_type: str
        ) -> list[list]:
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
                        In(entity=e, coordenate=c, shape_str=self.shape_str)
                    )
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
                        )
                    )
                elif next_coord == self.uncertainty:
                    transition_sentences.append(
                        From(
                            entity=entity,
                            coordenate=prev_coord,
                            shape_str=self.shape_str,
                        )
                    )
                else:
                    transition_sentences.append(
                        random.choice(
                            [
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
                            ]
                        )
                    )
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


class ComplexTrackingRequest(BaseModel):
    answer: Any
    d0: Any
    d1: Any
    d2: Any

    def get_question(self):
        pass

    def get_answer(self):
        pass


class ObjectInLocationPolar(ComplexTrackingRequest):
    answer: Literal["yes", "no", "unknown"]
    d0: Optional[Any] = None
    d1: Optional[Any] = None
    d2: Optional[Any] = None

    def get_question(self):
        return f"Is {self.d2.name} in the {self.d0.name}?"

    def get_answer(self):
        return self.answer


class ObjectInLocationWhat(ComplexTrackingRequest):
    answer: Literal["designated_object", "none", "unknown"]
    d0: Optional[Any] = None
    d1: Optional[Any] = None
    d2: Optional[Any] = None

    def get_question(self):
        return f"What is in the {self.d0.name}?"

    def get_answer(self):
        return self.answer


class ObjectInLocationWhere(ComplexTrackingRequest):
    answer: Literal["designated_location", "unknown"]
    d0: Optional[Any] = None
    d1: Optional[Any] = None
    d2: Optional[Any] = None

    def get_question(self):
        return f"Where is {self.d2.name}?"

    def get_answer(self):
        return self.answer


class ObjectsInLocation(BaseModel):
    dim0: list[Coordenate]
    dim1: list[Coordenate]
    dim2: list[Entity]

    @model_validator(mode="after")
    def _shuffle(self):
        random.shuffle(self.dim0)
        random.shuffle(self.dim1)
        random.shuffle(self.dim2)
        return self

    @property
    def as_tuple(self):
        return (self.dim0, self.dim1, self.dim2)


class ComplexTracking(BaseModel):
    model: Any
    states_qty: int
    topic: ComplexTrackingRequest
    uncertainty: list = None
    verbosity: Union[int, str] = Field(default=logging.INFO)
    logger: Optional[Any] = None
    log_file: Optional[Path] = None
    states: Optional[list[State]] = None
    deltas: Optional[Any] = None
    story: Optional[Story] = None
    fol: list[FOL] = None
    nl: list[str] = None
    num_transitions: int = 1
    dim0_obj_to_idx: Optional[dict] = None
    dim1_obj_to_idx: Optional[dict] = None
    dim2_obj_to_idx: Optional[dict] = None
    dim0_idx_to_obj: Optional[dict] = None
    dim1_idx_to_obj: Optional[dict] = None
    dim2_idx_to_obj: Optional[dict] = None
    shape: Optional[tuple[int, int, int]] = None
    location_to_locations_map: Optional[dict] = None
    shape_str: tuple
    p_antilocation: float = 0.5  # to be false, higher than
    location_matrix: Optional[Any] = None
    p_move_d2: float = 0.5  # to be true, lower than
    state_class: Optional[State] = None

    @model_validator(mode="after")
    def fill_logger(self):
        if not self.logger:
            self.logger = logger.get_logger(
                "ComplexTracking",
                level=self.verbosity,
                log_file=self.log_file,
            )
        return self

    @model_validator(mode="after")
    def check_shape_and_model(self):
        model_tuple = self.model.as_tuple
        if len(model_tuple) != len(self.shape_str):
            raise ValueError(
                f"Length mismatch: 'model.as_tuple()' has length {len(model_tuple)} "
                f"but 'shape_str' has length {len(self.shape_str)}."
            )
        return self

    def load_ontology_from_topic(self) -> Callable:
        # Define the mapping between answer types and loader functionsc
        loader_mapping: dict[type[ComplexTrackingRequest], Callable] = {
            ObjectInLocationPolar: self._object_in_location_polar,
            ObjectInLocationWhat: self._object_in_location_what,
            ObjectInLocationWhere: self._object_in_location_where,
        }
        uncertainty_mapping: dict[type[ComplexTrackingRequest], tuple] = {
            ObjectInLocationPolar: (
                Coordenate(name="nowhere"),
                Coordenate(name="nobody"),
                Entity(name="nothing"),
            ),
            ObjectInLocationWhat: (
                None,
                Coordenate(name="nobody"),
                Entity(name="nothing"),
            ),
            ObjectInLocationWhere: (
                None,
                Coordenate(name="nobody"),
                Entity(name="nothing"),
            ),
        }
        state_mapping: dict[type[ComplexTrackingRequest], State] = {
            ObjectInLocationPolar: ObjectInLocationStatePolar,
            ObjectInLocationWhat: ObjectInLocationState,
            ObjectInLocationWhere: ObjectInLocationState,
        }
        # Get the type of the answer
        answer_type = type(self.topic)
        if answer_type not in loader_mapping:
            raise ValueError(
                f"Unsupported answer type: {answer_type.__name__}. "
                f"Should be one of {[cls.__name__ for cls in loader_mapping]}"
            )
        # Set the uncertainty based on the answer type
        if uncertainty_mapping[answer_type]:
            self.uncertainty = uncertainty_mapping[answer_type]

        if state_mapping[answer_type]:
            self.state_class = state_mapping[answer_type]
        return loader_mapping[answer_type]

    def _create_aux(self):
        self.shape = (len(self.model.dim0), len(self.model.dim1), len(self.model.dim2))
        self.dim0_obj_to_idx = {o: i for i, o in enumerate(self.model.dim0)}
        self.dim1_obj_to_idx = {o: i for i, o in enumerate(self.model.dim1)}
        self.dim2_obj_to_idx = {o: i for i, o in enumerate(self.model.dim2)}
        self.dim0_idx_to_obj = {i: o for i, o in enumerate(self.model.dim0)}
        self.dim1_idx_to_obj = {i: o for i, o in enumerate(self.model.dim1)}
        self.dim2_idx_to_obj = {i: o for i, o in enumerate(self.model.dim2)}
        return

    def create_ontology(self):
        f_ontology = self.load_ontology_from_topic()
        self.states = f_ontology()
        # self.create_transitions()

    def initialize_state(self, i: int, condition: Callable) -> State:
        """
        Initializes the state for an entity in a location based on a given condition.
        Args:
            i (int): An integer identifier for the state.
            condition (Callable): A callable that takes a set of entities and returns a
            boolean indicating
                                  whether the condition is met.
        Returns:
            State: A initialized state that meets the given condition.
        """

        self.logger.info(
            "initialize_state:",
            i=i,
            answer=self.topic.answer,
        )
        s = self.create_random_state(i)
        t = 0
        while not condition(s.am):
            s = self.create_random_state(i)
            t += 1

        s.logger.info("State initialized", state=s, answer=self.topic.answer, i=i)
        return s

    def create_random_state(self, i: int) -> State:
        """
        Creates a random state for entities in coordinates with three dimensions.
        Args:
            i (int): The index to be assigned to the generated state.
        Returns:
            State: A state having as a 3D adjacency matrix,
            in sparse format (DOK).
        """
        self.logger.debug("Creating Random State", shape=self.shape)
        # Step 1: List of  objects
        objects = np.arange(self.shape[2] - 1)
        # Step 2: Pick N actors (can be repeated)
        num_objects = len(objects)
        actor_choices = np.random.choice(self.shape[1], num_objects, replace=True)
        # Step 3: Assign locations for the chosen actors
        unique_actors = np.unique(actor_choices)
        location_choices = np.random.choice(
            self.shape[0], len(unique_actors), replace=True
        )
        # Create a mapping of actor -> location to ensure one location per actor
        actor_to_location = dict(zip(unique_actors, location_choices))

        sparse_matrix = DOK(shape=self.shape, dtype=bool, fill_value=0)
        for obj, actor in zip(objects, actor_choices):
            loc = actor_to_location[actor]  # Ensure the actor gets a unique location
            sparse_matrix[loc, actor, obj] = 1

        # Get actors with nothing
        AWN = np.where(
            (sparse_matrix[:, :, :-1] == 0).to_coo().sum(axis=(0, 2)).todense()
            == sparse_matrix.shape[0] * (sparse_matrix.shape[2] - 1)
        )[0]
        AWN = list(AWN)
        # no for each actor with nothing, pick a random place, and give then the
        # `nothing`(-1) object
        if AWN:
            for a in AWN:
                loc = np.random.choice(sparse_matrix.shape[0])
                sparse_matrix[loc, a, -1] = 1
        s = self.state_class(
            am=sparse_matrix, index=i, verbosity=self.verbosity, log_file=self.log_file
        )
        return s

    def _object_in_location_polar(self):
        d0 = self.model.dim0[0]
        d1 = self.model.dim1[0]
        d2 = self.model.dim2[0]
        self.topic.d0 = d0
        self.topic.d1 = d1
        self.topic.d2 = d2
        self.model.dim0.append(self.uncertainty[0])
        self.model.dim1.append(self.uncertainty[1])
        self.model.dim2.append(self.uncertainty[2])
        self._create_aux()
        self.logger.info(
            "Creating _object_in_location_polar",
            topic=type(self.topic).__name__,
            answer=self.topic.answer,
            l=d0.name,
            a=d1.name,
            o=d2.name,
        )
        states = [None] * self.states_qty

        if self.topic.answer == "yes":
            i = self.states_qty - 1
            condition = lambda x: sum(x[0, :, 0]) == 1
            states[i] = self.initialize_state(i, condition)
            for j in list(reversed(range(i))):
                condition = lambda x: True
                # chose between move and object, or move an anctor
                axis = 2 if self.choice() else 1
                states[j] = self.create_new_state(j, states[j + 1], condition, axis)

        elif self.topic.answer == "no":
            if random.choice([0, 1]):
                # case for d2 not in d1
                i = self.states_qty - 1
                condition = lambda x: sum(x[0, :, 0]) == 0 and sum(x[-1, :, 0]) == 0
                states[i] = self.initialize_state(i, condition)
                for j in list(reversed(range(i))):
                    condition = lambda x: True
                    axis = 2 if self.choice() else 1
                    states[j] = self.create_new_state(j, states[j + 1], condition, axis)
            else:
                # case where d2 is in d0 = uncertinty, but previously was in d0.
                i = random.randint(0, self.states_qty - 2)
                condition = lambda x: sum(x[0, :, 0]) == 1
                states[i] = self.initialize_state(i, condition)
                # Backward
                for j in list(reversed(range(i))):
                    condition = lambda x: True
                    # chose between move and object, or move an anctor
                    axis = 2 if self.choice() else 1
                    states[j] = self.create_new_state(j, states[j + 1], condition, axis)
                # Forward
                # This mean, that the d2 remains in d0=uncertainty
                for j in range(i + 1, len(states)):
                    condition = lambda x: sum(x[-1, :, 0]) == 1
                    axis = 2 if self.choice() else 1
                    states[j] = self.create_new_state(j, states[j - 1], condition, axis)

        elif self.topic.answer == "unknown":
            if random.choice([0, 1]):
                # case where d3 always remain in nowhere (d3=-1)
                i = 0
                condition = lambda x: sum(x[-1, :, 0]) == 1
                self.logger.debug(
                    "Creating polar unknown, with 0", answer=self.topic.answer, i=i
                )
                states[i] = self.initialize_state(i, condition)
                for j in range(1, self.states_qty):
                    # remain always in nowhere
                    condition = lambda x: sum(x[-1, :, 0]) == 1
                    # chose between move and object, or move an actor
                    axis = 2 if self.choice() else 1
                    states[j] = self.create_new_state(j, states[j - 1], condition, axis)
            else:
                # case where d3 was not in d1, and neither in nowhere in certain point.
                # (this mean it was in some dim1 != d1)
                # and it can't be in nobody.
                i = random.randint(0, self.states_qty - 2)
                self.logger.debug(
                    "Creating polar unknown, with 0", answer=self.topic.answer, i=i
                )
                #
                condition = (
                    lambda x: sum(x[0, :, 0]) == 0
                    and sum(x[-1, :, 0]) == 0
                    and x[:, :-1, 0].to_coo().sum() == 1
                )  # This is an extra, so someone has to be in the same place
                states[i] = self.initialize_state(i, condition)
                self.logger.debug("Begin of backward")
                for j in list(reversed(range(i))):
                    # if j is the first iteration then,
                    # define a `filter` lambda function
                    condition = lambda x: True
                    axis = 2 if self.choice() else 1
                    states[j] = self.create_new_state(j, states[j + 1], condition, axis)
                # create the states after i, that where d2 remains in nowhere (d0==-1)
                self.logger.debug("Begin of forward")
                print("len(states)", len(states))
                for j in range(i + 1, len(states)):
                    condition = lambda x: sum(x[-1, :, 0]) == 1
                    # This sould force to pick the desired object to move
                    # to `nowhere` in the first iteration.
                    if j == i + 1:  # noqa: SIM108
                        filter = lambda x: x[:, :, [0]] == 1
                    else:
                        filter = None
                    axis = 2 if self.choice() else 1
                    states[j] = self.create_new_state(
                        j, states[j - 1], condition, axis, filter
                    )
        else:
            raise ValueError("Invalid answer value, should be 'yes' no 'no'")

        return states

    def initialize_state_with_antilocations(self, i: int, condition: Callable) -> State:
        """
        Initializes the state for an entity in a location based on a given condition.
        Args:
            i (int): An integer identifier for the state.
            condition (Callable): A callable that takes a set of entities and returns a
            boolean indicating
                                  whether the condition is met.
        Returns:
            State: A initialized state that meets the given condition.
        """

        self.logger.info(
            "initialize_state:",
            i=i,
            answer=self.topic.answer,
        )
        s = self.create_random_state_with_antilocations(i)
        t = 0
        while not condition(s.am):
            s = self.create_random_state_with_antilocations(i)
            t += 1

        s.logger.info("State initialized", state=s, answer=self.topic.answer, i=i)
        return s

    def choice_known_location(self):
        return np.random.uniform(0, 1) < self.p_antilocation

    def create_random_state_with_antilocations(self, i: int) -> State:
        """
        Creates a random state for entities in coordinates with three dimensions.
        Args:
            i (int): The index to be assigned to the generated state.
        Returns:
            State: A state having as a 3D adjacency matrix,
            in sparse format (DOK).
        """
        self.logger.debug("Creating Random State", shape=self.shape)
        # Step 1: List of  objects
        objects = np.arange(self.shape[2] - 1)
        # Step 2: Pick N actors (can be repeated)
        num_objects = len(objects)
        num_locations = self.shape[0] // 2
        actor_choices = np.random.choice(self.shape[1], num_objects, replace=True)
        unique_actors = np.unique(actor_choices)
        # conver unique actors to list of int
        unique_actors = unique_actors.tolist()
        # Step 3: Assign locations for the chosen actors
        location_choices = []
        for a in unique_actors:
            if self.choice_known_location():
                i_l = random.randint(0, num_locations)
            else:
                i_l = random.randint(num_locations, self.location_matrix.shape[0] - 1)
            vector = self.location_matrix[i_l]
            idx = np.where(vector == 1)
            location_choices.append(idx[0].tolist())

        # Create a mapping of actor -> location to ensure one location per actor
        actor_to_location = dict(zip(unique_actors, location_choices))
        sparse_matrix = DOK(shape=self.shape, dtype=bool, fill_value=0)
        for obj, actor in zip(objects, actor_choices):
            # get the location for the actor
            # list_loc = actor_to_location[actor]
            list_loc = np.atleast_1d(actor_to_location[actor])
            for loc in list_loc:
                sparse_matrix[loc, actor, obj] = 1

        # Get actors with nothing
        AWN = np.where(
            (sparse_matrix[:, :, :-1] == 0).to_coo().sum(axis=(0, 2)).todense()
            == sparse_matrix.shape[0] * (sparse_matrix.shape[2] - 1)
        )[0]
        AWN = list(AWN)
        # no for each actor with nothing, pick a random place, and give then the
        # `nothing`(-1) object
        if AWN:
            for a in AWN:
                if self.choice_known_location():
                    i_l = random.randint(0, num_locations)
                else:
                    i_l = random.randint(
                        num_locations, self.location_matrix.shape[0] - 1
                    )
                list_loc = np.where(vector == 1)[0].tolist()
                for loc in list_loc:
                    sparse_matrix[loc, a, -1] = 1
        s = self.state_class(
            am=sparse_matrix, index=i, verbosity=self.verbosity, log_file=self.log_file
        )
        return s

    def create_transition_map(
        self,
    ):
        """
        Given a matrix with possible locations for actors, this function return a dictionary
        where:
        - the keys are the origen of a transition
        - the values are a list of possible destinations.
        """
        n_l = self.location_matrix.shape[1] // 2
        location_with_allowed_actor_transitions = [
            np.where(row)[0].tolist() for row in self.location_matrix[: n_l * 2]
        ]
        first_half = location_with_allowed_actor_transitions[:n_l]
        second_half = location_with_allowed_actor_transitions[n_l:]

        # for the first half
        map = {}
        for f_i in first_half:
            # all first half except the current one
            map[tuple(f_i)] = [i for i in first_half if i != f_i]
        dict_second_half = {}
        for i_l, s_i in enumerate(second_half):
            # only the corresponding in the first half
            dict_second_half[tuple(s_i)] = [first_half[i_l]]
        # create a variable with the joined dictionaries
        map.update(dict_second_half)
        return map

    def _object_in_location_what(self):
        d0 = self.model.dim0[0]
        d1 = self.model.dim1[0]
        d2 = self.model.dim2[0]
        self.topic.d0 = d0
        self.topic.d1 = d1
        self.topic.d2 = d2
        # for dim0, due to anti locations, i need to add each element again
        # to the list, BUT, adding the 'anti-' prefix in each element name
        anti_locations = [Coordenate(name=f"anti-{d.name}") for d in self.model.dim0]
        self.model.dim0.extend(anti_locations)
        self.model.dim1.append(self.uncertainty[1])
        self.model.dim2.append(self.uncertainty[2])
        self._create_aux()
        self.location_matrix = operators.generate_location_matrix(self.shape[0] // 2)
        self.location_to_locations_map = self.create_transition_map()
        self.logger.info(
            "Creating _object_in_location_polar",
            topic=type(self.topic).__name__,
            answer=self.topic.answer,
            l=d0.name,
            a=d1.name,
            o=d2.name,
            location_matrix_MB="{:.2f} MB".format(
                self.location_matrix.nbytes / 1024 / 1024
            ),
        )

        states = [None] * self.states_qty
        i = self.states_qty - 1
        # What-Questions: A question of the form “What is in l?” or similar.
        if self.topic.answer == "designated_object":

            def check_designated_object(x) -> bool:
                n_l = x.shape[0] // 2
                in_designated_location = x[:, :, 0].to_coo().T[0][0] == 1
                f_half = x[:, :, 0].to_coo().T[0][:n_l]
                s_half = x[:, :, 0].to_coo().T[0][n_l:]
                # validate that vector of ones is generated.
                valid_designated_ok = list((f_half + s_half).todense()) == ([1] * n_l)
                # Fot the corresponding anti-location, for all the other objects (1:-1)
                # there should be certainty that they are in the anti-location.
                current_in_not_l = np.array(
                    x[n_l, :, 1:-1].to_coo().sum(axis=0).todense()
                )
                ideal_not_in_l = np.array([1] * (x.shape[2] - 2))
                answer_others_not_in_l = np.array_equal(
                    current_in_not_l, ideal_not_in_l
                )
                # None of the others objects (1:-1) can be in the full-nowhere location
                # Thas's why (n_l:).
                not_object_in_full_nowhere = np.all(
                    x[n_l:, :, 1:-1].to_coo().sum(axis=0) < n_l
                )
                return (
                    in_designated_location
                    and valid_designated_ok
                    and answer_others_not_in_l
                    and not_object_in_full_nowhere
                )

            condition = check_designated_object
        if self.topic.answer == "none":

            def check_none(x) -> bool:
                n_l = x.shape[0] // 2
                is_empty_location = x[0, :, :-1].to_coo().sum() == 0
                # For the anti-location designated, all Os added by column, are there.
                current_not_in_l = np.array(
                    x[n_l, :, :-1].to_coo().sum(axis=0).todense()
                )
                ideal_not_in_l = np.array([1] * (x.shape[2] - 1))
                all_not_in_l = np.array_equal(current_not_in_l, ideal_not_in_l)
                # None objects in the full-nowhere location
                not_object_in_full_nowhere = np.all(
                    x[n_l:, :, :-1].to_coo().sum(axis=0) < n_l
                )
                return is_empty_location and all_not_in_l and not_object_in_full_nowhere

            condition = check_none
        if self.topic.answer == "unknown":

            def check_unknown(x) -> bool:
                n_locations = x.shape[0] // 2
                is_empty_location = x[0, :, :-1].to_coo().sum() == 0
                results = []

                # Loop over known locations (skipping index 0)
                for loc in range(1, n_locations):
                    x_known = x[loc, :, :-1]
                    # Loop over all anti-location
                    # (except the first anti-location, for abvious reasons)
                    for anti_idx in range(n_locations + 1, x.shape[0]):
                        # Skip the anti-location paired with the current known location
                        if anti_idx == loc + n_locations:
                            continue
                        mat_sum = (x_known + x[anti_idx, :, :-1]).todense()
                        # Check if any element in the sum is less than n_locations - 1
                        results.append(np.any(mat_sum < n_locations - 1))

                return np.any(results) and is_empty_location

            condition = check_unknown

        states[i] = self.initialize_state_with_antilocations(i, condition)
        # for j in list(reversed(range(i))):
        #    condition = lambda x: True
        #    axis = 2 if self.choice() else 1
        #    states[j] = self.create_new_state(j, states[j + 1], condition, axis)
        return states

    def _object_in_location_where(self):
        raise NotImplementedError("Not implemented yet")

    def create_new_state(
        self,
        j: int,
        state: State,
        condition: Callable,
        axis: int,
        filter: Optional[Callable] = None,
    ) -> State:
        """
        Create a new state for an entity in a location based on the current state and
        the given conditions.
        Args:
            j (int): An identifier for the state.
            state (State): The current state of derive a new one.
            condition (Callable): A callable that represents a condition to meet by
            the transition.
        Returns:
            State: The new state of the entity in the location after
            applying the transitions.
        """

        new_am = state.create_transition(self.num_transitions, condition, axis, filter)
        if new_am is None:
            self.logger.error(
                "Fail both: make_actor_transition & make_object_transition"
            )
            raise ValueError(
                "There could;t be found any compatible solution for axis / transition"
            )
        new_state = self.state_class(
            am=new_am, index=j, verbosity=self.verbosity, log_file=self.log_file
        )
        return new_state

    def choice(self):
        return np.random.uniform(0, 1) < self.p_move_d2

    def create_transitions(self):
        def process_delta(diff):
            """
            This function generate the delta in the form of:
            [(entity, coord), [origin], [end]].
            First it checks if for the data there is more than one location,
            if its the case, then it means that the transition is an actor_transition,
            if not, then its a object_transition.
            For object_transition, it removes the `nothing` object self.shape[2]-1
            from the data and coords.
            And preserve the -1 and 1, to know who is the origin and who is the end.
            For the actor_transition, as object tracking is not important,
            only need to locate whichs is the origin location,
            and which is the end location.

            Args:
                diff (COO): The difference between two states.

            Returns:
                list: A list of tuples with the following format:
                    [(entity, coord), [origin], [end]].

            """
            # check if all happend in the same location,
            # if its not, then its a actor_transition
            q_locations = len(np.unique(diff.coords[0]))
            if q_locations == 1:
                # get the nothing object to remove then.
                idx_to_kept = diff.coords.T[:, -1] != (self.shape[2] - 1)
                # remove -1&1 from data w.r.t to nothing
                diff.data = diff.data.T[idx_to_kept].T
                # remove -1 from coords w.r.t to nothing
                diff.coords = diff.coords.T[idx_to_kept].T
                pair = diff.data
                # get (origin,end)
                if pair[0] == -1:
                    o = 0
                    e = 1
                else:
                    o = 1
                    e = 0
                delta_j = [(2, 1), diff.coords.T[o][1:], diff.coords.T[e][1:]]
                return delta_j
            elif q_locations == 2:
                # due to i dont care abount the items, just pick the place of -1 and 1
                pair = diff.data
                e = np.where(pair == 1)[0][0]
                o = np.where(pair == -1)[0][0]
                delta_j = [(1, 0), diff.coords.T[o][:-1], diff.coords.T[e][:-1]]
                return delta_j
            else:
                raise ValueError("q_locations should be 1 or 2")

        deltas = []
        for i in range(0, self.states_qty - 1):
            current_state, reference_state = (
                self.states[i + 1].am,
                self.states[i].am,
            )
            diff = current_state.to_coo() - reference_state.to_coo()
            transition_info = process_delta(diff)
            self.logger.info("Transition", i=i, transition=transition_info)
            deltas.append(transition_info)
        self.deltas = deltas
        return

    def create_fol(self):
        def enumerate_model(element: list, shape_type: str) -> list[list]:
            enumeration = []
            for e in element:
                if e != self.uncertainty:
                    enumeration.append(Exists(thing=e, shape_str=shape_type))
            return enumeration

        def describe_states(state) -> list[list]:
            state_sentences = []
            # 1) Actors with nothing in a place.
            actors_in_locations_nothing = state.am[:-1, :-1, -1]
            for loc, a in actors_in_locations_nothing.data:
                state_sentences.append(
                    In(
                        entity=self.dim1_idx_to_obj[a],
                        coordenate=self.dim0_idx_to_obj[loc],
                        shape_str=(self.shape_str[0], self.shape_str[1]),
                    )
                )
            # 2) Actor with objects in a place
            actors_in_locations_objects = state.am[:, :-1, :-1]
            actor_done = []
            for loc, a, o in actors_in_locations_objects.data:
                if a not in actor_done:
                    actor_done.append(a)
                    # check that the l is not the uncertainty
                    if loc != self.shape[0] - 1:
                        state_sentences.append(
                            In(
                                entity=self.dim1_idx_to_obj[a],
                                coordenate=self.dim0_idx_to_obj[loc],
                                shape_str=(self.shape_str[0], self.shape_str[1]),
                            )
                        )
                state_sentences.append(
                    In(
                        entity=self.dim2_idx_to_obj[o],
                        coordenate=self.dim1_idx_to_obj[a],
                        shape_str=(self.shape_str[1], self.shape_str[2]),
                    )
                )

            # 3) Objects in a place
            objects_in_locations = state.am[:-1, -1, :-1]
            for loc, o in objects_in_locations.data:
                state_sentences.append(
                    In(
                        entity=self.dim2_idx_to_obj[o],
                        coordenate=self.dim0_idx_to_obj[loc],
                        shape_str=(self.shape_str[0], self.shape_str[2]),
                    )
                )
            # Regarding actos in nowhere, there is no sentece.
            return state_sentences

        def describe_transitions(state: State) -> list[list]:
            i = state.index
            delta = self.deltas[i]

            # Define mapping for different delta cases
            delta_mappings = {
                (1, 0): (
                    self.dim1_idx_to_obj,
                    self.dim0_idx_to_obj,
                    ("Location", "Actor"),
                    self.uncertainty[0],
                ),
                (2, 1): (
                    self.dim2_idx_to_obj,
                    self.dim1_idx_to_obj,
                    ("Actor", "Object"),
                    self.uncertainty[1],
                ),
            }

            if delta[0] not in delta_mappings:
                raise ValueError("Invalid delta")

            entity_map, coord_map, shape_str, uncertainty = delta_mappings[delta[0]]

            idx_entity = delta[1][1]
            idx_prev_coord = delta[1][0]
            idx_next_coord = delta[2][0]

            entity = entity_map[idx_entity]
            prev_coord = coord_map[idx_prev_coord]
            next_coord = coord_map[idx_next_coord]

            if prev_coord == uncertainty:
                transition_sentences = To(
                    entity=entity, coordenate=next_coord, shape_str=shape_str
                )
            elif next_coord == uncertainty:
                transition_sentences = From(
                    entity=entity, coordenate=prev_coord, shape_str=shape_str
                )
            else:
                transition_sentences = random.choice(
                    [
                        To(entity=entity, coordenate=next_coord, shape_str=shape_str),
                        FromTo(
                            entity=entity,
                            coordenate1=prev_coord,
                            coordenate2=next_coord,
                            shape_str=shape_str,
                        ),
                    ]
                )

            return [transition_sentences]

        world_enumerate = []
        story = []

        for t, dim_str in zip(self.model.as_tuple, self.shape_str):
            world_enumerate.extend(enumerate_model(t[:-1], dim_str))
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
