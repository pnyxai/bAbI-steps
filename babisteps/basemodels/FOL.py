import random
from typing import Union

from pydantic import BaseModel

from babisteps.basemodels.nodes import Coordenate, Entity


class FOL(BaseModel):
    shape_str: tuple

    def to_nl(self):
        pass


class Exists(FOL):
    thing: Union[Entity, Coordenate]
    shape_str: str

    def to_nl(self):
        if self.shape_str == "Location":
            return f"There is a {self.thing.name}."
        elif self.shape_str == "Actor":
            return f"{self.thing.name} is present."
        elif self.shape_str == "Object":
            return f"There is a {self.thing.name}."


class In(FOL):
    entity: Union[Entity, Coordenate]
    coordenate: Union[Entity, Coordenate]

    def to_nl(self) -> str:
        e, c = self.entity.name, self.coordenate.name

        if self.shape_str == ("Location", "Actor"):
            return f"{e} is in the {c}."
        elif self.shape_str == ("Location", "Object"):
            return f"The {e} is in the {c}."
        elif self.shape_str == ("Actor", "Object"):
            options = [
                f"{c} has the {e}.",
                f"{c} is carrying the {e}.",
            ]
            return random.choice(options)
        else:
            raise ValueError("Invalid types for In relation")


class To(FOL):
    entity: Union[Entity, Coordenate]
    coordenate: Union[Entity, Coordenate]

    def to_nl(self) -> str:
        e, c = self.entity.name, self.coordenate.name

        if self.shape_str == ("Location", "Actor"):
            options = [
                f"{e} went to the {c}.",
                f"{e} traveled to the {c}.",
                f"{e} entered the {c}.",
                f"{e} reached to the {c}.",
                f"{e} moved to the {c}.",
            ]
            return random.choice(options)

        elif self.shape_str == ("Object", "Location"):
            options = [
                f"The {e} was carried to the {c}.",
                f"The {e} was taken to the {c}.",
                f"The {e} was moved to the {c}.",
            ]
            return random.choice(options)

        elif self.shape_str == ("Actor", "Object"):
            options = [
                f"{c} took the {e}.",
                f"{c} grabbed the {e}.",
                f"{c} picked the {e}.",
            ]
            return random.choice(options)
        else:
            raise ValueError("Invalid types for To relation")


class From(FOL):
    entity: Union[Entity, Coordenate]
    coordenate: Union[Entity, Coordenate]

    def to_nl(self) -> str:
        e, c = self.entity.name, self.coordenate.name
        if self.shape_str == ("Location", "Actor"):
            options = [
                f"{e} left the {c}.",
                f"{e} abandoned the {c}.",
            ]
            return random.choice(options)

        elif self.shape_str == ("Object", "Location"):
            options = [
                f"The {e} was carried from the {c}.",
                f"The {e} was taken from the {c}.",
                f"The {e} was moved from the {c}.",
            ]
            return random.choice(options)

        elif self.shape_str == ("Actor", "Object"):
            options = [
                f"{c} left the {e}.",
                f"{c} dropped the {e}.",
                f"{c} abandoned the {e}.",
            ]
            return random.choice(options)
        else:
            raise ValueError("Invalid types for From relation")


class FromTo(FOL):
    entity: Union[Entity, Coordenate]
    coordenate1: Union[Entity, Coordenate]
    coordenate2: Union[Entity, Coordenate]

    def to_nl(self) -> str:
        e, c1, c2 = (
            self.entity.name,
            self.coordenate1.name,
            self.coordenate2.name,
        )

        if self.shape_str == ("Location", "Actor"):
            options = [
                f"{e} went from the {c1} to the {c2}.",
                f"{e} traveled from the {c1} to the {c2}.",
                f"{e} entered from the {c1} to the {c2}.",
                f"{e} reached from the {c1} to the {c2}.",
                f"{e} moved from the {c1} to the {c2}.",
            ]
            return random.choice(options)

        elif self.shape_str == ("Object", "Location"):
            options = [
                f"The {e} was carried from the {c1} to the {c2}.",
                f"The {e} was taken from the {c1} to the {c2}.",
                f"The {e} was moved from the {c1} to the {c2}.",
            ]
            return random.choice(options)
        elif self.shape_str == ("Actor", "Object"):
            options = [
                f"{c1} gave the {e} to {c2}.",
                f"{c1} passed the {e} to {c2}.",
                f"{c1} gave the {e} to {c2}.",
            ]
            return random.choice(options)
        else:
            raise ValueError("Invalid types for FromTo relation")
