import random
from typing import Any, Callable, Literal, Optional, get_type_hints

from pydantic import BaseModel, model_validator

from babisteps.basemodels.generators import (
    ACTORS_NONE_ANSWERS, DELIM, OBJECTS_LOCATION_EVENT_NONE_ANSWERS,
    UNKNONW_ANSWERS, SimpleTrackerBaseGenerator)
from babisteps.basemodels.nodes import Coordenate, Entity

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
        if self.answer == "yes" or self.answer == "no":
            return [self.answer]
        elif self.answer == "unknown":
            return UNKNONW_ANSWERS


class ActorInLocationWho(SimpleTrackerRequest):
    answer: Literal["designated_entity", "none", "unknown"]

    def get_question(self):
        return f"Who is in the {self.coordenate.name}?"

    def get_answer(self):
        if self.answer == "designated_entity":
            return [self.entity.name]
        elif self.answer == "none":
            return ACTORS_NONE_ANSWERS
        elif self.answer == "unknown":
            return UNKNONW_ANSWERS
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
            return [self.coordenate.name]
        elif self.answer == "unknown":
            return UNKNONW_ANSWERS
        else:
            raise ValueError(
                "Invalid answer value, should be 'designated_location' or 'unknown'"
            )


class ActorWithObjectPolar(SimpleTrackerRequest):
    answer: Literal["yes", "no"]

    def get_question(self):
        return f"Has {self.coordenate.name} the {self.entity.name}?"

    def get_answer(self):
        return [self.answer]


class ActorWithObjectWhat(SimpleTrackerRequest):
    answer: Literal["designated_object", "none"]

    def get_question(self):
        return f"What has {self.coordenate.name}?"

    def get_answer(self) -> list[str]:
        if self.answer == "designated_object":
            return [self.entity.name]
        elif self.answer == "none":
            return OBJECTS_LOCATION_EVENT_NONE_ANSWERS
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
            return [self.coordenate.name]
        elif self.answer == "none":
            return ACTORS_NONE_ANSWERS
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


class SimpleTracker(SimpleTrackerBaseGenerator):
    topic: SimpleTrackerRequest

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

    def generate(self):
        self.create_ontology()
        self.create_fol()

    def get_json(self):
        json = self.story.create_json()
        # TODO (Nicolas):
        # Options should be taken randomly from the "none" and "unknown" list of anwers
        options = list(get_type_hints(self.topic)["answer"].__args__)
        if isinstance(self.topic,
                      (ActorInLocationPolar, ActorWithObjectPolar)):
            pass
        elif isinstance(self.topic, ActorInLocationWho):
            options.remove("designated_entity")
            options.extend([e.name for e in self.model.entities])
            options.remove("none")
            options.append(random.choice(ACTORS_NONE_ANSWERS))
        elif isinstance(self.topic, ActorInLocationWhere):
            options.remove("designated_location")
            options.extend([c.name for c in self.model.coordenates])
            options.remove("nowhere")
        elif isinstance(self.topic, ActorWithObjectWhat):
            options.remove("designated_object")
            options.extend([e.name for e in self.model.entities])
            options.remove("none")
            options.append(random.choice(OBJECTS_LOCATION_EVENT_NONE_ANSWERS))
        elif isinstance(self.topic, ActorWithObjectWho):
            options.remove("designated_actor")
            options.extend([c.name for c in self.model.coordenates])
            options.remove("none")
            options.remove("nobody")
            options.append(random.choice(ACTORS_NONE_ANSWERS))
        else:
            raise ValueError("Invalid answer type")

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

    def get_txt(self):
        return self.story.create_txt()
