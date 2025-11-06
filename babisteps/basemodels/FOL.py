import random
from abc import ABC, abstractmethod
from typing import Union

from pydantic import BaseModel

from babisteps.basemodels.nodes import (Coordinate, Entity, Relationship,
                                        TemporalTrackingEvent)


class FOL(BaseModel, ABC):
    shape_str: tuple

    @abstractmethod
    def to_nl(self):
        """Abstract method to be implemented in subclasses."""
        pass


class Exists(FOL):
    thing: Union[Entity, Coordinate]
    shape_str: str

    def to_nl(self):
        if self.shape_str == "locations":
            return f"There is a {self.thing.name}."
        elif self.shape_str == "actors":
            return f"{self.thing.name} is present."
        elif self.shape_str == "objects":
            return f"There is a {self.thing.name}."
        elif self.shape_str == "events":
            return f"There was a {self.thing.name}."
        else:
            raise ValueError("Invalid type for Exists relation")


class In(FOL):
    entity: Union[Entity, Coordinate]
    coordinate: Union[Entity, Coordinate]

    def to_nl(self) -> str:
        e, c = self.entity.name, self.coordinate.name

        if self.shape_str == ("locations", "actors"):
            return f"{e} is in the {c}."
        elif self.shape_str == ("locations", "objects"):
            return f"The {e} is in the {c}."
        elif self.shape_str == ("actors", "objects"):
            options = [
                f"{c} has the {e}.",
                f"{c} is carrying the {e}.",
            ]
            return random.choice(options)
        else:
            raise ValueError("Invalid types for In relation")


class To(FOL):
    entity: Union[Entity, Coordinate]
    coordinate: Union[Entity, Coordinate]

    def to_nl(self) -> str:
        e, c = self.entity.name, self.coordinate.name

        if self.shape_str == ("locations", "actors"):
            options = [
                f"{e} went to the {c}.",
                f"{e} traveled to the {c}.",
                f"{e} entered the {c}.",
                f"{e} reached to the {c}.",
                f"{e} moved to the {c}.",
            ]
            return random.choice(options)

        elif self.shape_str == ("objects", "locations"):
            options = [
                f"The {e} was carried to the {c}.",
                f"The {e} was taken to the {c}.",
                f"The {e} was moved to the {c}.",
            ]
            return random.choice(options)

        elif self.shape_str == ("actors", "objects"):
            options = [
                f"{c} took the {e}.",
                f"{c} grabbed the {e}.",
                f"{c} picked the {e}.",
            ]
            return random.choice(options)
        else:
            raise ValueError("Invalid types for To relation")


class From(FOL):
    entity: Union[Entity, Coordinate]
    coordinate: Union[Entity, Coordinate]

    def to_nl(self) -> str:
        e, c = self.entity.name, self.coordinate.name
        if self.shape_str == ("locations", "actors"):
            options = [
                f"{e} left the {c}.",
                f"{e} abandoned the {c}.",
            ]
            return random.choice(options)

        elif self.shape_str == ("objects", "locations"):
            options = [
                f"The {e} was carried from the {c}.",
                f"The {e} was taken from the {c}.",
                f"The {e} was moved from the {c}.",
            ]
            return random.choice(options)

        elif self.shape_str == ("actors", "objects"):
            options = [
                f"{c} left the {e}.",
                f"{c} dropped the {e}.",
                f"{c} abandoned the {e}.",
            ]
            return random.choice(options)
        else:
            raise ValueError("Invalid types for From relation")


class FromTo(FOL):
    entity: Union[Entity, Coordinate]
    coordinate1: Union[Entity, Coordinate]
    coordinate2: Union[Entity, Coordinate]

    def to_nl(self) -> str:
        e, c1, c2 = (
            self.entity.name,
            self.coordinate1.name,
            self.coordinate2.name,
        )

        if self.shape_str == ("locations", "actors"):
            options = [
                f"{e} went from the {c1} to the {c2}.",
                f"{e} traveled from the {c1} to the {c2}.",
                f"{e} entered from the {c1} to the {c2}.",
                f"{e} reached from the {c1} to the {c2}.",
                f"{e} moved from the {c1} to the {c2}.",
            ]
            return random.choice(options)

        elif self.shape_str == ("objects", "locations"):
            options = [
                f"The {e} was carried from the {c1} to the {c2}.",
                f"The {e} was taken from the {c1} to the {c2}.",
                f"The {e} was moved from the {c1} to the {c2}.",
            ]
            return random.choice(options)
        elif self.shape_str == ("actors", "objects"):
            options = [
                f"{c1} gave the {e} to {c2}.",
                f"{c1} passed the {e} to {c2}.",
                f"{c1} gave the {e} to {c2}.",
            ]
            return random.choice(options)
        else:
            raise ValueError("Invalid types for FromTo relation")


class Out(FOL):
    entity: Union[Entity, Coordinate]
    coordinate: Union[Entity, Coordinate]

    def to_nl(self) -> str:
        e, c = self.entity.name, self.coordinate.name

        if self.shape_str == ("locations", "actors"):
            options = [
                f"{e} is away from the {c}.",
                f"{e} is outside of the {c}.",
            ]
            return random.choice(options)

        elif self.shape_str == ("locations", "objects"):
            options = [
                f"The {e} is away from the {c}.",
                f"The {e} is outside of the {c}.",
            ]
            return random.choice(options)


class IsRelated(FOL):
    relation: Relationship
    entity0: Union[Entity]
    entity1: Union[Entity]

    def to_nl(self):
        e0, e1 = self.entity0.name, self.entity1.name
        if self.shape_str in [("locations", ), ("objects", )]:
            # chose if use base or opposite relation
            options = [
                f"The {e0} is {random.choice(self.relation.base)} the {e1}.",
                f"The {e1} is {random.choice(self.relation.opposite)} the {e0}.",
            ]
            return random.choice(options)
        elif self.shape_str == ("actors", ):
            # chose if use base or opposite relation
            options = [
                f"{e0} is {random.choice(self.relation.base)} {e1}.",
                f"{e1} is {random.choice(self.relation.opposite)} {e0}.",
            ]
            return random.choice(options)
        elif self.shape_str == ("events", ):
            # chose if use base or opposite relation
            options = [
                f"The {e0} was {random.choice(self.relation.base)} the {e1}.",
                f"The {e1} was {random.choice(self.relation.opposite)} the {e0}.",
            ]
            return random.choice(options)
        else:
            raise ValueError("Invalid types for IsRelated relation")


class IsTemporalRelated(FOL):
    relation: Relationship
    event0: TemporalTrackingEvent
    event1: TemporalTrackingEvent

    def to_nl(self):
        e0, e1 = self.event0, self.event1
        if self.shape_str == ("locations", "actors"):
            # check if the event has the same entities in order to avoid repetition
            if e0.entity == e1.entity:
                options = [
                    f"{e0.entity.name} went to the {e0.coordinate.name} {random.choice(self.relation.base)} going to the {e1.coordinate.name}.",  # noqa: E501
                    f"{e0.entity.name} went to the {e1.coordinate.name} {random.choice(self.relation.opposite)} going to the {e0.coordinate.name}."  # noqa: E501
                ]
            else:
                options = [
                    f"{e0.entity.name} went to the {e0.coordinate.name} {random.choice(self.relation.base)} {e1.entity.name} went to the {e1.coordinate.name}.",  # noqa: E501
                    f"{e1.entity.name} went to the {e1.coordinate.name} {random.choice(self.relation.opposite)} {e0.entity.name} went to the {e0.coordinate.name}."  # noqa: E501
                ]

            return random.choice(options)
        else:
            raise ValueError("Invalid types for IsTemporalRelated relation")
