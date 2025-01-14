import logging
import random
from typing import Annotated, Any, Callable, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from babisteps.basemodels.FOL import FOL, Exists, From, FromTo, In, To
from babisteps.basemodels.nodes import (
    Coordenate,
    Entity,
    EntityInLocationState,
    State,
    UnitState,
)
from babisteps.utils import logger


class Generator(BaseModel):
    model: Any
    states_qty: int
    verbosity: Union[int, str] = Field(default=logging.INFO)
    logger: Optional[Any] = None
    # min_transitions: int = Field(ge=1)
    # max_transitions: int = Field(ge=1)
    states: Optional[list[State]] = None
    deltas: Optional[Any] = None
    fol: list[FOL] = None
    nl: list[str] = None

    @model_validator(mode="after")
    def fill_logger(self):
        if not self.logger:
            self.logger = logger.get_logger("Generator", level=self.verbosity)
        return self

    # @model_validator(mode="after")
    # def validate_max_transitions(self):
    #     if self.max_transitions < self.min_transitions:
    #         raise ValueError(
    #             "max_transitions should be greater or equal than min_transitions"
    #         )
    #     # it should be lower than the numbers of entities
    #     if self.max_transitions > len(self.model.entities):
    #         raise ValueError(
    #             "max_transitions should be lower than the number of entities")
    #     return self

    def create_ontology(self):
        pass

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

    def initialize_state(self, i: int, condition: Callable) -> EntityInLocationState:
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
        self.logger.debug("State initialized", answer=self.answer, i=i)
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
            element: Union[list[Entity], list[Coordenate]],
        ) -> list[list]:
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
                        In(entity=unit.entity, coordenate=unit.coordenate)
                    )
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
                        To(entity=entity, coordenate=next_coord)
                    )
                elif next_coord == self.uncertainty:
                    transition_sentences.append(
                        From(entity=entity, coordenate=prev_coord)
                    )
                else:
                    transition_sentences.append(
                        random.choice(
                            [
                                To(entity=entity, coordenate=next_coord),
                                FromTo(
                                    entity=entity,
                                    coordenate1=prev_coord,
                                    coordenate2=next_coord,
                                ),
                            ]
                        )
                    )
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


# -------------------------
# Model
# -------------------------


class EntitiesInCoordenates(BaseModel):
    entities: list[Entity]
    coordenates: list[Coordenate]
    entities_type: str
    coordenates_type: str


# -------------------------
# Answer
# -------------------------


class Answer(BaseModel):
    answer: Any


class Polar(Answer):
    answer: Literal["yes", "no", "nowhere", "nobody"]
    uncertainty: Literal["nowhere", "nobody"]


class Who(Answer):
    answer: Literal["designated_actor", "none", "unknown"]
    uncertainty: Literal["unknown_actor"]


class EntitiesInCoordenatesPolar(Generator):
    model: EntitiesInCoordenates
    answer: Annotated[int, Field(strict=True, ge=0, le=2)]
    states: Optional[list[UnitState]] = None
    deltas: Optional[list] = None
    uncertainty: Union[Coordenate, Entity] = None

    def create_ontology(self):
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
        self.model.coordenates.append(self.uncertainty)
        self.logger.info(
            "Creating EntitiesInCoordenatesPolar",
            answer=self.answer,
            e=e.name,
            c=c.name,
        )
        states = [None] * self.states_qty

        # Answer 1: a is in l
        if self.answer == 1:
            i = self.states_qty - 1
            condition = lambda x: (e, c) in x
            states[i] = self.initialize_state(i, condition)  # NewEntityInLocationState
            for j in list(reversed(range(i))):
                any_condition = lambda pair: True
                all_condition = lambda pair: True
                states[j] = self.create_new_state(
                    j, states[j + 1], any_condition, all_condition
                )

        # Answer 0: a is not in l
        if self.answer == 0:
            i = self.states_qty - 1
            condition = lambda x: (e, c) not in x and (e, self.uncertainty) not in x
            states[i] = self.initialize_state(i, condition)  # NewEntityInLocationState
            for j in list(reversed(range(i))):
                any_condition = lambda pair: True
                all_condition = lambda pair: True
                states[j] = self.create_new_state(
                    j, states[j + 1], any_condition, all_condition
                )

        if self.answer == 2:
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
                    states[j] = self.create_new_state(
                        j, states[j - 1], any_condition, all_condition
                    )
            else:
                i = random.randint(0, self.states_qty - 1)
                condition = lambda x: (e, c) in x
                states[i] = self.initialize_state(i, condition)
                for j in list(reversed(range(i))):
                    any_condition = lambda pair: True
                    all_condition = lambda pair: True
                    states[j] = self.create_new_state(
                        j, states[j + 1], any_condition, all_condition
                    )

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
                    states[j] = self.create_new_state(
                        j, states[j - 1], any_condition, all_condition
                    )

        self.states = states
        self.create_deltas()


class EntitiesInCoordenatesWho(Generator):
    model: EntitiesInCoordenates
    answer: Annotated[int, Field(strict=True, ge=0, le=2)]
    states: Optional[list[UnitState]] = None
    deltas: Optional[list] = None
    uncertainty: Union[Coordenate, Entity] = None

    def create_ontology(self):
        e = self.model.entities[0]
        c = self.model.coordenates[0]
        self.model.coordenates.append(self.uncertainty)
        self.logger.info(
            "Creating EntitiesInCoordenatesWho", answer=self.answer, e=e.name, c=c.name
        )
        states = [None] * self.states_qty

        if self.answer == 1:
            i = self.states_qty - 1
            condition = lambda x: (e, c) in x and all(
                (entity, c) not in x for entity in self.model.entities[1:]
            )
            states[i] = self.initialize_state(i, condition)  # NewEntityInLocationState
            for j in list(reversed(range(i))):
                any_condition = lambda pair: True
                all_condition = lambda pair: True
                states[j] = self.create_new_state(
                    j, states[j + 1], any_condition, all_condition
                )

        elif self.answer == 0:
            i = self.states_qty - 1

            condition = (
                lambda x: all((entity, c) not in x for entity in self.model.entities)
                and len(
                    [
                        (entity, self.uncertainty)
                        for entity in self.model.entities
                        if (entity, self.uncertainty) in x
                    ]
                )
                < self.states_qty
            )

            states[i] = self.initialize_state(i, condition)  # NewEntityInLocationState

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
                            j, states[j + 1], any_condition, all_condition
                        )
            else:
                self.logger.debug("There were not entities in uncertainty")
                for j in list(reversed(range(i))):
                    any_condition = lambda pair: True
                    all_condition = lambda pair: True
                    states[j] = self.create_new_state(
                        j, states[j + 1], any_condition, all_condition
                    )

        elif self.answer == 2:
            i = self.states_qty - 1
            # empty_l = all(A, lambda x : x \not \in l)
            empty_l = lambda x: all(
                (entity, c) not in x for entity in self.model.entities
            )
            # some_in_UN = any(A, lambda x : x \in NW)
            some_in_UN = lambda x: any(
                (entity, self.uncertainty) in x for entity in self.model.entities
            )
            condition = lambda x: empty_l(x) and some_in_UN(x)
            states[i] = self.initialize_state(i, condition)
            # ANW = list(choose({s.actorsInNowhere(A)}))
            EIU = states[i].get_entities_in_coodenate(self.uncertainty)
            for j in list(reversed(range(i))):
                ue = random.choice(self.model.entities)
                if ue in EIU:
                    any_condition = lambda pair: True
                    all_condition = lambda pair, ue=ue: pair != (ue, c) and \
                        pair != (ue,self.uncertainty)
                    EIU.remove(ue)
                else:
                    any_condition = lambda pair, ue=ue: pair[0] == ue
                    all_condition = lambda pair: True
                states[j] = self.create_new_state(
                    j, states[j + 1], any_condition, all_condition
                )
        else:
            raise ValueError("Invalid answer value")

        self.states = states
        self.create_deltas()
