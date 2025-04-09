import random
from typing import Any, Callable, Literal, Optional, Union, get_type_hints

import numpy as np
from pydantic import BaseModel, model_validator
from sparse._dok import DOK

from babisteps.basemodels.FOL import FOL, Exists, From, FromTo, In, To
from babisteps.basemodels.generators import BaseGenerator
from babisteps.basemodels.nodes import (Coordenate, Entity,
                                        EntityInCoordenateState, State)
from babisteps.basemodels.stories import Story

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
        return f"Who is in the {self.coordenate.name}?"

    def get_answer(self):
        if self.answer == "designated_entity":
            return self.entity.name
        elif self.answer == "none" or self.answer == "unknown":
            return self.answer
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
            return self.answer
        else:
            raise ValueError(
                "Invalid answer value, should be 'designated_location' or 'unknown'"
            )


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
        elif self.answer == "none":
            return self.answer
        else:
            raise ValueError(
                "Invalid answer value, should be 'designated_object' or 'none'"
            )


class ActorWithObjectWho(SimpleTrackerRequest):
    answer: Literal["designated_actor", "none"]

    def get_question(self):
        return f"Who has the {self.entity.name}?"

    def get_answer(self):
        if self.answer == "designated_actor":
            return self.coordenate.name
        elif self.answer == "none":
            return self.answer
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


class SimpleTracker(BaseGenerator):
    model: Any
    states_qty: int
    topic: SimpleTrackerRequest
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
        topic_type = type(self.topic)

        if topic_type not in loader_mapping:
            raise ValueError(
                f"Unsupported answer type: {topic_type.__name__}. "
                f"Should be one of {[cls.__name__ for cls in loader_mapping]}")
        # Set the uncertainty based on the answer type
        if uncertainty_mapping[topic_type]:
            self.uncertainty = uncertainty_mapping[topic_type]

        return loader_mapping[topic_type]

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
                    states[j] = self.create_new_state(j, states[j + 1],
                                                      condition)
            else:
                # case where e is in uncertinty, but previously was in c.
                i = random.randint(0, self.states_qty - 2)
                condition = lambda x: x[0, 0] == 1
                states[i] = self.initialize_state(i, condition)
                for j in list(reversed(range(i))):
                    condition = lambda x: True
                    states[j] = self.create_new_state(j, states[j + 1],
                                                      condition)
                # create the states after i
                for j in range(i + 1, len(states)):
                    condition = lambda x: x[-1, 0] == 1
                    states[j] = self.create_new_state(j, states[j - 1],
                                                      condition)

        elif self.topic.answer == "unknown":
            if random.choice([0, 1]):
                i = 0
                condition = lambda x: x[-1, 0] == 1
                states[i] = self.initialize_state(i, condition)
                for j in range(1, self.states_qty):
                    condition = lambda x: x[-1, 0] == 1
                    states[j] = self.create_new_state(j, states[j - 1],
                                                      condition)
            else:
                i = random.randint(0, self.states_qty - 2)
                condition = lambda x: x[0, 0] == 0 and x[-1, 0] == 0
                states[i] = self.initialize_state(i, condition)
                for j in list(reversed(range(i))):
                    condition = lambda x: True
                    states[j] = self.create_new_state(j, states[j + 1],
                                                      condition)
                # create the states after i
                for j in range(i + 1, len(states)):
                    condition = lambda x: x[-1, 0] == 1
                    states[j] = self.create_new_state(j, states[j - 1],
                                                      condition)
        else:
            raise ValueError(
                "Invalid answer value, should be 'yes', 'no' or 'unknown'")

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
            self.logger.debug(
                "Creating _actor_in_location_who with answer none")
            i = self.states_qty - 1
            condition = lambda x: sum(x[0, :]) == 0 and sum(x[-1, :]
                                                            ) < self.states_qty
            states[i] = self.initialize_state(i, condition)

            EIU = states[i].get_entities_in_coodenate(
                self.c2idx[self.uncertainty])

            if EIU:
                self.logger.debug(
                    "Entities in uncertainty",
                    EIU=EIU,
                )
                while EIU:
                    EIU = states[i].get_entities_in_coodenate(
                        self.c2idx[self.uncertainty])
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
                            condition = lambda x, EIU=EIU: all(x[
                                -1, EIU].todense() == [1] * len(EIU))
                        states[j] = self.create_new_state(
                            j, states[j + 1], condition)
            else:
                self.logger.debug("There were not entities in uncertainty")
                for j in list(reversed(range(i))):
                    condition = lambda x: True
                    states[j] = self.create_new_state(j, states[j + 1],
                                                      condition)

        elif self.topic.answer == "unknown":
            i = self.states_qty - 1
            empty_l = lambda x: sum(x[0, :]) == 0
            some_in_UN = lambda x: sum(x[-1, :]) > 0

            condition = lambda x: empty_l(x) and some_in_UN(x)
            states[i] = self.initialize_state(i, condition)
            EIU = states[i].get_entities_in_coodenate(
                self.c2idx[self.uncertainty])
            for j in list(reversed(range(i))):
                ue = random.choice(self.model.entities)
                x_ue = self.e2idx[ue]
                if x_ue in EIU:
                    condition = (lambda x, x_ue=x_ue: x[0, x_ue] == 0 and x[
                        -1, x_ue] == 0)
                    EIU.remove(x_ue)
                else:
                    condition = lambda x: all(x[-1, EIU].todense() == [1] *
                                              len(EIU))
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
            raise ValueError(
                "Invalid answer value, should be 1 (YES) or 0 (NO)")

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
                "Invalid answer value, should be 'designated_actor' or 'none'")

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

    def generate(self):
        self.create_ontology()
        self.create_fol()

    def get_json(self):
        json = self.story.create_json()
        options = list(get_type_hints(self.topic)['answer'].__args__)
        if isinstance(self.topic,
                      (ActorInLocationPolar, ActorWithObjectPolar)):
            pass
        elif isinstance(self.topic, ActorInLocationWho):
            options.remove("designated_entity")
            options.extend([e.name for e in self.model.entities])
        elif isinstance(self.topic, ActorInLocationWhere):
            options.remove("designated_location")
            options.extend([c.name for c in self.model.coordenates])
        elif isinstance(self.topic, ActorWithObjectWhat):
            options.remove("designated_object")
            options.extend([e.name for e in self.model.entities])
        elif isinstance(self.topic, ActorWithObjectWho):
            options.remove("designated_actor")
            options.extend([c.name for c in self.model.coordenates])
        else:
            raise ValueError("Invalid answer type")

        random.shuffle(options)
        json['options'] = options

        if self.name:
            json['leaf'] = self.name.split('_-_')[0]
            json['leaf_label'] = self.name.split('_-_')[1]
            json['leaf_index'] = self.name.split('_-_')[2]

        return json

    def get_txt(self):
        return self.story.create_txt()
