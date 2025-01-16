import logging
import random
from typing import Any, Callable, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from babisteps.basemodels.FOL import FOL, Exists, From, FromTo, In, To
from babisteps.basemodels.nodes import (Coordenate, Entity,
                                        EntityInLocationState, State,
                                        UnitState)
from babisteps.utils import logger

# -------------------------
# Answer
# -------------------------


class TopicRequest(BaseModel):
    answer: Any
    entity: Optional[Entity] = None
    coordenate: Optional[Coordenate] = None

    def get_question(self):
        pass

    def get_answer(self):
        pass


class ActorInLocationPolar(TopicRequest):
    answer: Literal["yes", "no", "unknown"]

    def get_question(self):
        return f"Is {self.entity.name} in the {self.coordenate.name}?"

    def get_answer(self):
        return self.answer


class ActorInLocationWho(TopicRequest):
    answer: Literal["designated_entity", "none", "unknown"]

    def get_question(self):
        return f"Who is in {self.coordenate.name}?"

    def get_answer(self):
        if self.answer == "designated_entity":
            return self.entity.name
        elif self.answer == "none":
            return "None"
        elif self.answer == "unknown":
            return "Unknown"
        else:
            raise ValueError(
                "Invalid answer, should be 'designated_entity', 'none' or 'unknown'"
            )


class ActorInLocationWhere(TopicRequest):
    answer: Literal["designated_location", "unknown"]

    def get_question(self):
        return f"Where is {self.entity.name}?"

    def get_answer(self):
        if self.answer == "designated_location":
            return self.coordenate.name
        elif self.answer == "unknown":
            return "unknown"


class ActorWithObjectPolar(TopicRequest):
    answer: Literal["yes", "no"]

    def get_question(self):
        return f"Has {self.coordenate.name} the {self.entity.name}?"

    def get_answer(self):
        return self.answer


class ActorWithObjectWhat(TopicRequest):
    answer: Literal["designated_object", "none"]

    def get_question(self):
        return f"What has {self.coordenate.name}?"

    def get_answer(self):
        if self.answer == "designated_object":
            return self.entity.name
        else:
            return "None"


class ActorWithObjectWho(TopicRequest):
    answer: Literal["designated_actor", "none"]

    def get_question(self):
        return f"Who has the {self.entity.name}?"

    def get_answer(self):
        if self.answer == "designated_actor":
            return self.coordenate.name
        elif self.answer == "none":
            return "None"
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
    entities_type: str
    coordenates_type: str

    @model_validator(mode="after")
    def _shuffle(self):
        random.shuffle(self.entities)
        random.shuffle(self.coordenates)
        return self


class SimpleTracker(BaseModel):
    model: Any
    states_qty: int
    topic: TopicRequest
    uncertainty: Optional[Coordenate] = None
    verbosity: Union[int, str] = Field(default=logging.INFO)
    logger: Optional[Any] = None
    states: Optional[list[State]] = None
    deltas: Optional[Any] = None
    fol: list[FOL] = None
    nl: list[str] = None

    @model_validator(mode="after")
    def fill_logger(self):
        if not self.logger:
            self.logger = logger.get_logger("Generator", level=self.verbosity)
        return self

    def load_ontology_from_topic(self) -> Callable:
        # Define the mapping between answer types and loader functions
        loader_mapping: dict[type[TopicRequest], Callable] = {
            ActorInLocationPolar: self._actor_in_location_polar,
            ActorInLocationWho: self._actor_in_location_who,
            ActorInLocationWhere: self._actor_in_location_where,
            ActorWithObjectPolar: self._actor_with_object_polar,
            ActorWithObjectWhat: self._actor_with_object_what,
            ActorWithObjectWho: self._actor_with_object_who,
        }
        uncertainty_mapping: dict[type[TopicRequest], Coordenate] = {
            ActorInLocationPolar: Coordenate(name="nowhere", type="Location"),
            ActorInLocationWho: Coordenate(name="nowhere", type="Location"),
            ActorInLocationWhere: Coordenate(name="nowhere", type="Location"),
            ActorWithObjectPolar: None,
            ActorWithObjectWhat: None,
            ActorWithObjectWho: Coordenate(name="nobody", type="Actor"),
        }

        # Get the type of the answer
        answer_type = type(self.topic)

        if answer_type not in loader_mapping:
            raise ValueError(
                f"Unsupported answer type: {answer_type.__name__}. "
                f"Should be one of {[cls.__name__ for cls in loader_mapping]}")
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
        self.logger.info(
            "Creating _actor_in_location_polar",
            answer=self.topic.answer,
            e=e.name,
            c=c.name,
        )
        states = [None] * self.states_qty
        # Answer 1: a is in l
        if self.topic.answer == "yes":
            i = self.states_qty - 1
            condition = lambda x: (e, c) in x
            states[i] = self.initialize_state(
                i, condition)  # NewEntityInLocationState
            for j in list(reversed(range(i))):
                any_condition = lambda pair: True
                all_condition = lambda pair: True
                states[j] = self.create_new_state(j, states[j + 1],
                                                  any_condition, all_condition)

        # Answer 0: a is not in l
        elif self.topic.answer == "no":
            i = self.states_qty - 1
            condition = lambda x: (e, c) not in x and (e, self.uncertainty
                                                       ) not in x
            states[i] = self.initialize_state(
                i, condition)  # NewEntityInLocationState
            for j in list(reversed(range(i))):
                any_condition = lambda pair: True
                all_condition = lambda pair: True
                states[j] = self.create_new_state(j, states[j + 1],
                                                  any_condition, all_condition)

        elif self.topic.answer == "unknown":
            if random.choice([0, 1]):
                i = 0
                condition = lambda x: (e, self.uncertainty) in x
                states[i] = self.initialize_state(i, condition)
                for j in range(1, self.states_qty):
                    any_condition = lambda pair: True
                    all_condition = (
                        lambda pair: pair[0] != e
                    )  # all entities should be different from a, due it will be always
                    # in self.uncertainty
                    states[j] = self.create_new_state(j, states[j - 1],
                                                      any_condition,
                                                      all_condition)
            else:
                i = random.randint(0, self.states_qty - 1)
                # a \not \in uncertintie && e \not in c
                condition = lambda x: (e, self.uncertainty) not in x and (
                    e, c) not in x
                states[i] = self.initialize_state(i, condition)
                for j in list(reversed(range(i))):
                    any_condition = lambda pair: True
                    all_condition = lambda pair: True
                    states[j] = self.create_new_state(j, states[j + 1],
                                                      any_condition,
                                                      all_condition)

                # create the states after i
                for j in range(i + 1, len(states)):
                    if j == i + 1:  # Here we place the entity in the self.uncertainty.
                        any_condition = lambda pair: pair == (
                            e,
                            self.uncertainty,
                        )
                        all_condition = lambda pair: True
                    else:
                        # Now there should be any ocurrence of entity
                        any_condition = lambda pair: True
                        all_condition = lambda pair: pair[0] != e
                    states[j] = self.create_new_state(j, states[j - 1],
                                                      any_condition,
                                                      all_condition)
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
        self.logger.info(
            "Creating _actor_in_location_who",
            answer=self.topic.answer,
            e=e.name,
            c=c.name,
        )
        states = [None] * self.states_qty

        if self.topic.answer == "designated_entity":
            i = self.states_qty - 1
            condition = lambda x: (e, c) in x and all(
                (entity, c) not in x for entity in self.model.entities[1:])
            states[i] = self.initialize_state(
                i, condition)  # NewEntityInLocationState
            for j in list(reversed(range(i))):
                any_condition = lambda pair: True
                all_condition = lambda pair: True
                states[j] = self.create_new_state(j, states[j + 1],
                                                  any_condition, all_condition)

        elif self.topic.answer == "none":
            self.logger.debug(
                "Creating _actor_in_location_who with answer none")
            i = self.states_qty - 1

            condition = (lambda x: all(
                (entity, c) not in x for entity in self.model.entities
            ) and len([(entity, self.uncertainty)
                       for entity in self.model.entities
                       if (entity, self.uncertainty) in x]) < self.states_qty)

            states[i] = self.initialize_state(
                i, condition)  # NewEntityInLocationState

            EIU = states[i].get_entities_in_coodenate(self.uncertainty)

            if EIU:
                self.logger.debug(
                    "Entities in uncertainty",
                    EIU=[entity.name for entity in EIU],
                )
                while EIU:
                    EIU = states[i].get_entities_in_coodenate(self.uncertainty)
                    for j in list(reversed(range(i))):
                        ue = random.choice(self.model.entities)
                        if ue in EIU:
                            self.logger.debug(
                                "Placing entity from NW to coordenate c",
                                entity=ue.name,
                                coordenate=c.name,
                            )
                            any_condition = lambda pair, ue=ue: pair == (ue, c)
                            all_condition = lambda pair: True
                            EIU.remove(ue)
                        else:
                            any_condition = lambda pair, ue=ue: pair[0] == ue
                            all_condition = lambda pair: True

                        states[j] = self.create_new_state(
                            j, states[j + 1], any_condition, all_condition)
            else:
                self.logger.debug("There were not entities in uncertainty")
                for j in list(reversed(range(i))):
                    any_condition = lambda pair: True
                    all_condition = lambda pair: True
                    states[j] = self.create_new_state(j, states[j + 1],
                                                      any_condition,
                                                      all_condition)

        elif self.topic.answer == "unknown":
            i = self.states_qty - 1
            # empty_l = all(A, lambda x : x \not \in l)
            empty_l = lambda x: all(
                (entity, c) not in x for entity in self.model.entities)
            # some_in_UN = any(A, lambda x : x \in NW)
            some_in_UN = lambda x: any((entity, self.uncertainty) in x
                                       for entity in self.model.entities)
            condition = lambda x: empty_l(x) and some_in_UN(x)
            states[i] = self.initialize_state(i, condition)
            # ANW = list(choose({s.actorsInNowhere(A)}))
            EIU = states[i].get_entities_in_coodenate(self.uncertainty)
            for j in list(reversed(range(i))):
                ue = random.choice(self.model.entities)
                if ue in EIU:
                    any_condition = lambda pair: True
                    all_condition = lambda pair, ue=ue: pair != (
                        ue, c) and pair != (
                            ue,
                            self.uncertainty,
                        )
                    EIU.remove(ue)
                else:
                    any_condition = lambda pair, ue=ue: pair[0] == ue
                    all_condition = lambda pair: True
                states[j] = self.create_new_state(j, states[j + 1],
                                                  any_condition, all_condition)
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
        # this time, entitie can be nowhere as DESIGNATED_LOCATION)
        c = self.model.coordenates[0]
        self.model.coordenates.append(self.uncertainty)
        self.topic.entity = e
        self.topic.coordenate = c
        self.logger.info(
            "Creating _actor_in_location_where",
            answer=self.topic.answer,
            e=e.name,
            c=c.name,
        )
        states = [None] * self.states_qty

        i = self.states_qty - 1
        if self.topic.answer == "designated_location":
            condition = lambda x: (e, c) in x
        elif self.topic.answer == "unknown":
            condition = lambda x: (e, self.uncertainty) in x
        else:
            raise ValueError(
                "Invalid answer value, should be 'designated_location' or 'unknown'"
            )

        states[i] = self.initialize_state(i, condition)
        for j in list(reversed(range(i))):
            any_condition = lambda pair: True
            all_condition = lambda pair: True
            states[j] = self.create_new_state(j, states[j + 1], any_condition,
                                              all_condition)

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
        states = [None] * self.states_qty
        self.logger.info(
            "Creating _actor_with_object_polar",
            answer=self.topic.answer,
            e=e.name,
            c=c.name,
        )

        i = self.states_qty - 1
        if self.topic.answer == "yes":
            condition = lambda x: (e, c) in x
        elif self.topic.answer == "no":
            condition = lambda x: (e, c) not in x and (e, self.uncertainty
                                                       ) not in x
        else:
            raise ValueError(
                "Invalid answer value, should be 1 (YES) or 0 (NO)")

        states[i] = self.initialize_state(i, condition)
        for j in list(reversed(range(i))):
            any_condition = lambda pair: True
            all_condition = lambda pair: True
            states[j] = self.create_new_state(j, states[j + 1], any_condition,
                                              all_condition)

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
        states = [None] * self.states_qty
        i = self.states_qty - 1

        # case answer == DESIGNATED_OBJECT:
        if self.topic.answer == "designated_object":
            # only_o_in_a = o \in a && all(O[1:], lambda x : x \not \in a)
            condition = lambda x: (e, c) in x and all(
                (entity, c) not in x for entity in self.model.entities[1:])
        # case answer == NO_OBJECT:
        elif self.topic.answer == "none":
            # empty_a = all(O, lambda x : x \not \in a)
            condition = lambda x: all(
                (entity, c) not in x for entity in self.model.entities)
        else:
            raise ValueError(
                "Invalid answer value, should be 1 (DESIGNATED_OBJECT) or 0 (NO_OBJECT)"
            )

        states[i] = self.initialize_state(i, condition)
        for j in list(reversed(range(i))):
            any_condition = lambda pair: True
            all_condition = lambda pair: True
            states[j] = self.create_new_state(j, states[j + 1], any_condition,
                                              all_condition)

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
        self.topic.entity = e
        self.topic.coordenate = c
        states = [None] * self.states_qty
        i = self.states_qty - 1
        # Define condition based on the answer

        if self.topic.answer == "designated_actor":
            condition = lambda x: (e, c) in x and all(
                (entity, c) not in x for entity in self.model.entities[1:])
        elif self.topic.answer == "none":
            # entity o \in NB
            condition = lambda x: (e, self.uncertainty) in x
        else:
            raise ValueError(
                "Invalid answer value, should be 1 (DESIGNATED_ACTOR) or 0 (NONE)"
            )

        states[i] = self.initialize_state(i, condition)
        for j in list(reversed(range(i))):
            any_condition = lambda pair: True
            all_condition = lambda pair: True
            states[j] = self.create_new_state(j, states[j + 1], any_condition,
                                              all_condition)

        self.logger.info(
            "_actor_with_object_who successfully created",
            answer=self.topic.answer,
            e=e.name,
            c=c.name,
        )

        return states

    def create_ontology(self):
        f_ontology = self.load_ontology_from_topic()
        self.states = f_ontology()
        self.create_deltas()

    def create_new_state(
        self,
        j: int,
        state: UnitState,
        any_condition: Callable,
        all_condition: Callable,
    ) -> EntityInLocationState:
        """
        Create a new state for an entity in a location based on the current state and
        the given conditions.
        Args:
            j (int): An identifier for the state.
            state (UnitState): The current state of the entity in the location.
            any_condition (Callable): A callable that represents a condition to meet for
            any transition.
            all_condition (Callable): A callable that represents a condition to meet
            for all transitions.
        Returns:
            EntityInLocationState: The new state of the entity in the location after
            applying the transitions.
        """
        # num_transitions = random.randint(self.min_transitions,
        #                                 self.max_transitions)
        #
        num_transitions = 1
        delta = state.create_delta(
            num_transitions,
            self.model.coordenates,
            any_condition,
            all_condition,
        )
        self.logger.debug("Delta", i=j, delta=delta)
        new_state = state.create_state_from_delta(j, delta)
        return new_state

    def initialize_state(self, i: int,
                         condition: Callable) -> EntityInLocationState:
        """
        Initializes the state for an entity in a location based on a given condition.
        Args:
            i (int): An integer identifier for the state.
            condition (Callable): A callable that takes a set of entities and returns a
            boolean indicating
                                  whether the condition is met.
        Returns:
            EntityInLocationState: The initialized state that meets the given condition.
        """

        self.logger.info("Creating Answer:", i=i)
        s = self.create_random_state(i)
        t = 0
        while not condition(s.attr_as_set):
            self.logger.debug("Condition not met", i=i, state=s)
            s = self.create_random_state(i)
            t += 1

        self.logger.debug("State initialized",
                          state=s,
                          answer=self.topic.answer,
                          i=i)
        return s

    def create_random_state(self, i: int) -> EntityInLocationState:
        """
        Creates a random state for entities in coordenates.
        Args:
            i (int): The index to be assigned to the generated state.
        Returns:
            EntityInLocationState: A state object containing entities and their
            randomly assigned coordenates.
        """

        entities = self.model.entities
        coordenates = [random.choice(self.model.coordenates) for _ in entities]
        entity_coord_pairs = list(zip(entities, coordenates))
        u_s = []
        for pair in entity_coord_pairs:
            e, c = pair[0], pair[1]
            u_i = UnitState(entity=e, coordenate=c)
            u_s.append(u_i)
        s = EntityInLocationState(attr=u_s, index=i)

        return s

    def create_deltas(self):
        deltas = []
        for i in range(0, self.states_qty - 1):
            current_state, reference_state = (
                self.states[i + 1].attr_as_set,
                self.states[i].attr_as_set,
            )
            d = current_state.difference(reference_state)
            deltas.append(d)
        self.deltas = deltas

    def create_fol(self):

        def enumerate_model(
            element: Union[list[Entity], list[Coordenate]], ) -> list[list]:
            enumeration = []
            for e in element:
                if e != self.uncertainty:
                    enumeration.append(Exists(thing=e))
            return enumeration

        def describe_states(state: State) -> list[list]:
            state_sentences = []
            for unit in state.attr:
                if unit.coordenate != self.uncertainty:
                    # state_sentences.append(["In", unit.entity.name, unit.coordenate])
                    state_sentences.append(
                        In(entity=unit.entity, coordenate=unit.coordenate))
            return state_sentences

        def describre_transitions(state: State) -> list[list]:
            i = state.index
            delta = self.deltas[i]
            transition_sentences = []
            for d in delta:
                prev_coord = state.get_entity_coordenate(d[0])
                entity, next_coord = d[0], d[1]

                if prev_coord == self.uncertainty:
                    transition_sentences.append(
                        To(entity=entity, coordenate=next_coord))
                elif next_coord == self.uncertainty:
                    transition_sentences.append(
                        From(entity=entity, coordenate=prev_coord))
                else:
                    transition_sentences.append(
                        random.choice([
                            To(entity=entity, coordenate=next_coord),
                            FromTo(
                                entity=entity,
                                coordenate1=prev_coord,
                                coordenate2=next_coord,
                            ),
                        ]))
            return transition_sentences

        sentences = []
        # get each attribute in self.model and itereate over it
        sentences.extend(enumerate_model(self.model.entities))
        sentences.extend(enumerate_model(self.model.coordenates))
        sentences.extend(describe_states(self.states[0]))
        for s in self.states[0:-1]:
            sentences.extend(describre_transitions(s))
        self.fol = sentences

    def create_nl(self):
        self.nl = [f.to_nl() for f in self.fol]

    def print_transition(self):
        self.logger.info("Initial state", state=self.states[0])
        for i, d in enumerate(self.deltas):
            self.logger.info("Delta", i=i, delta=d)
        self.logger.info("Final state", state=self.states[-1])
