import random
from typing import Union

from pydantic import BaseModel

from babisteps.basemodels.nodes import Coordenate, Entity


class FOL(BaseModel):

    def to_nl(self):
        pass


class Exists(FOL):
    thing: Union[Entity, Coordenate]

    def to_nl(self):
        if self.thing.type == "Location":
            return f"There is a {self.thing.name}"
        elif self.thing.type == "Actor":
            return f"{self.thing.name} is present"
        elif self.thing.type == "Object":
            return f"There is a {self.thing.name}"


class In(FOL):
    entity: Entity
    coordenate: Coordenate

    def to_nl(self) -> str:
        e_t, c_t = self.entity.type, self.coordenate.type
        e, c = self.entity.name, self.coordenate.name

        if e_t == "Actor" and c_t == "Location":
            return f"{e} is in the {c}"
        elif e_t == "Object" and c_t == "Location":
            return f"The {e} is in the {c}"
        elif e_t == "Object" and c_t == "Actor":
            options = [
                f"{c} has the {e}.",
                f"{c} is carrying the {e}.",
            ]
            return random.choice(options)
        else:
            raise ValueError("Invalid types for In relation")


class To(FOL):
    entity: Entity
    coordenate: Coordenate

    def to_nl(self) -> str:
        e_t, c_t = self.entity.type, self.coordenate.type
        e, c = self.entity.name, self.coordenate.name

        if e_t == "Actor" and c_t == "Location":
            options = [
                f"{e} went to the {c}.",
                f"{e} traveled to the {c}.",
                f"{e} entered the {c}.",
                f"{e} reached to the {c}.",
                f"{e} moved to the {c}.",
            ]
            return random.choice(options)

        elif e_t == "Object" and c_t == "Location":
            options = [
                f"The {e} was carried to the {c}.",
                f"The {e} was taken to the {c}.",
                f"The {e} was moved to the {c}.",
            ]
            return random.choice(options)

        elif e_t == "Object" and c_t == "Actor":
            options = [
                f"{c} took the {e}.",
                f"{c} grabbed the {e}.",
                f"{c} picked the {e}.",
            ]
            return random.choice(options)
        else:
            raise ValueError("Invalid types for To relation")


class From(FOL):
    entity: Entity
    coordenate: Coordenate

    def to_nl(self) -> str:
        e_t, c_t = self.entity.type, self.coordenate.type
        e, c = self.entity.name, self.coordenate.name
        if e_t == "Actor" and c_t == "Location":
            options = [
                f"{e} left the {c}.",
                f"{e} abandoned the {c}.",
            ]
            return random.choice(options)

        elif e_t == "Object" and c_t == "Location":
            options = [
                f"The {e} was carried from the {c}.",
                f"The {e} was taken from the {c}.",
                f"The {e} was moved from the {c}.",
            ]
            return random.choice(options)

        elif e_t == "Object" and c_t == "Actor":
            options = [
                f"{c} left the {e}.",
                f"{c} dropped the {e}.",
                f"{c} abandoned the {e}.",
            ]
            return random.choice(options)
        else:
            raise ValueError("Invalid types for From relation")


class FromTo(FOL):
    entity: Entity
    coordenate1: Coordenate
    coordenate2: Coordenate

    def to_nl(self) -> str:
        e_t, c1_t, c2_t = (
            self.entity.type,
            self.coordenate1.type,
            self.coordenate2.type,
        )
        e, c1, c2 = (
            self.entity.name,
            self.coordenate1.name,
            self.coordenate2.name,
        )

        if e_t == "Actor" and c1_t == "Location" and c2_t == "Location":
            options = [
                f"{e} went from the {c1} to the {c2}.",
                f"{e} traveled from the {c1} to the {c2}.",
                f"{e} entered from the {c1} to the {c2}.",
                f"{e} reached from the {c1} to the {c2}.",
                f"{e} moved from the {c1} to the {c2}.",
            ]
            return random.choice(options)

        elif e_t == "Object" and c1_t == "Location" and c2_t == "Location":
            options = [
                f"The {e} was carried from the {c1} to the {c2}.",
                f"The {e} was taken from the {c1} to the {c2}.",
                f"The {e} was moved from the {c1} to the {c2}.",
            ]
            return random.choice(options)
        elif e_t == "Object" and c1_t == "Actor" and c2_t == "Actor":
            options = [
                f"{c1} gave the {e} to {c2}.",
                f"{c1} passed the {e} to {c2}.",
                f"{c1} gave the {e} to {c2}.",
            ]
            return random.choice(options)
        else:
            raise ValueError("Invalid types for FromTo relation")
