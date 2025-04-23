import random
from itertools import combinations
from typing import Any, Callable, Literal, Optional, Union

from pydantic import BaseModel, model_validator

from babisteps.basemodels.generators import DELIM, SimpleTrackerBaseGenerator
from babisteps.basemodels.nodes import Coordenate, Entity

# -------------------------
# Answer
# -------------------------


class ListingRequest(BaseModel):
    answer: Any
    entities: Optional[list[Entity]] = None
    coordenate: Optional[Coordenate] = None

    def get_question(self):
        pass

    def get_answer(self):
        pass


class ActorInLocationWho(ListingRequest):
    answer: Union[int, Literal["none", "unknown"]]

    @model_validator(mode='after')
    def validate_answer(self):
        if isinstance(self.answer, int) and self.answer <= 1:
            raise ValueError(
                "If answer is an integer, it must be greater than 1")
        return self

    def get_question(self):
        return f"Who is in the {self.coordenate.name}?"

    def get_answer(self):
        if isinstance(self.answer, int):
            return sorted([e.name for e in self.entities])
        elif self.answer == "none" or self.answer == "unknown":
            return [self.answer]
        else:
            raise ValueError(
                "Invalid answer, should be 'designated_entities', 'none' or 'unknown'"
            )


class ActorWithObjectWhat(ListingRequest):
    answer: Union[int, Literal["none"]]

    @model_validator(mode='after')
    def validate_answer(self):
        if isinstance(self.answer, int) and self.answer <= 1:
            raise ValueError(
                "If answer is an integer, it must be greater than 1")
        return self

    def get_question(self):
        return f"What has {self.coordenate.name}?"

    def get_answer(self):
        if isinstance(self.answer, int):
            return sorted([e.name for e in self.entities])
        elif self.answer == "none" or self.answer == "unknown":
            return [self.answer]
        else:
            raise ValueError(
                "Invalid answer, should be 'designated_entities', 'none' or 'unknown'"
            )


class Listing(SimpleTrackerBaseGenerator):
    topic: ListingRequest

    def load_ontology_from_topic(self) -> Callable:
        # Define the mapping between answer types and loader functions
        loader_mapping: dict[type[ListingRequest], Callable] = {
            ActorInLocationWho: self._actor_in_location_who,
            ActorWithObjectWhat: self._actor_with_object_what,
        }
        uncertainty_mapping: dict[type[ListingRequest], Coordenate] = {
            ActorInLocationWho: Coordenate(name="nowhere"),
            ActorWithObjectWhat: None,
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

    def get_random_combinations(self, n):
        """
        Returns n random combinations (subsets) of the input list.
        The empty set and the full set are included in the possible combinations.
        Order doesn't matter.
        
        Args:
            string_list: List of strings
            n: Number of random combinations to return
            
        Returns:
            List of n random combinations (as lists)
        """
        string_list = [e.name for e in self.model.entities]
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

    def _actor_in_location_who(self):
        c = self.model.coordenates[0]
        self.topic.coordenate = c
        if isinstance(self.topic.answer, int):
            e = self.model.entities[:self.topic.answer]
            self.topic.entities = e
        self.model.coordenates.append(self.uncertainty)
        self._create_aux()
        self.logger.info(
            "Creating _actor_in_location_who",
            answer=self.topic.answer,
            e=([e.name for e in self.topic.entities]
               if self.topic.entities else self.topic.answer),
            c=c.name,
        )
        states = [None] * self.states_qty

        if isinstance(self.topic.answer, int):
            q = self.topic.answer
            i = self.states_qty - 1
            condition = lambda x: sum(x[0, :q]) == q and sum(x[0, q:]) == 0
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
            e=([e.name for e in self.topic.entities]
               if self.topic.entities else self.topic.answer),
            c=c.name,
        )
        return states

    def _actor_with_object_what(self):
        c = self.model.coordenates[0]
        self.topic.coordenate = c
        if isinstance(self.topic.answer, int):
            e = self.model.entities[:self.topic.answer]
            self.topic.entities = e
        self.model.coordenates.append(self.uncertainty)
        self._create_aux()
        self.logger.info(
            "Creating _actor_with_object_what",
            answer=self.topic.answer,
            e=([e.name for e in self.topic.entities]
               if self.topic.entities else self.topic.answer),
            c=c.name,
        )
        states = [None] * self.states_qty

        i = self.states_qty - 1
        if isinstance(self.topic.answer, int):
            q = self.topic.answer
            condition = lambda x: sum(x[0, :q]) == q and sum(x[0, q:]) == 0
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
            "_actor_with_object_what successfully created:",
            answer=self.topic.answer,
            e=([e.name for e in self.topic.entities]
               if self.topic.entities else self.topic.answer),
            c=c.name,
        )
        return states

    def generate(self):
        self.create_ontology()
        self.create_fol()

    def get_json(self):
        json = self.story.create_json()
        if isinstance(self.topic, ActorInLocationWho):
            options = [["none"], ["unknown"]]
        elif isinstance(self.topic, ActorWithObjectWhat):
            options = [["none"]]
        else:
            raise ValueError("Invalid answer type")
        o = self.get_random_combinations(n=6)
        if isinstance(self.topic.answer, int):
            if self.topic.entities is None:
                raise ValueError(
                    "self.topic.entities is None, cannot derive 'anws'")
            anws = sorted([e.name for e in self.topic.entities])
            # if is not in the options, add it
            if anws not in options:
                options.append(anws)
        options.extend(o)
        # Shuffle to avoid bias in the answer order
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
