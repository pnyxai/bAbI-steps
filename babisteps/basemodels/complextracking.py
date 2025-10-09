import random
from typing import Any, Callable, Literal, Optional, get_type_hints

import numpy as np
from pydantic import BaseModel, model_validator
from sparse import DOK

from babisteps import operators
from babisteps.basemodels import groups as gr
from babisteps.basemodels.FOL import FOL, Exists, From, FromTo, In, Out, To
from babisteps.basemodels.generators import (
    DELIM, OBJECTS_LOCATION_EVENT_NONE_ANSWERS, REPLACE_PLACEHOLDER,
    UNKNONW_ANSWERS, BaseGenerator)
from babisteps.basemodels.nodes import (Coordenate, Entity,
                                        ObjectInLocationState,
                                        ObjectInLocationStatePolar, State)
from babisteps.basemodels.stories import Story


class ComplexTrackingRequest(BaseModel):
    answer: Any
    d0: Any
    d1: Any
    d2: Any

    def get_question(self):
        pass

    def get_answer(self) -> list[str]:
        pass

    def get_reponse_tempalte(self):
        pass


class ObjectInLocationPolar(ComplexTrackingRequest):
    answer: Literal["yes", "no", "unknown"]
    d0: Optional[Any] = None
    d1: Optional[Any] = None
    d2: Optional[Any] = None

    def get_question(self):
        return f"Is the {self.d2.name} in the {self.d0.name}?"

    def get_answer(self) -> list[str]:
        if self.answer == "yes" or self.answer == "no":
            return [self.answer]
        elif self.answer == "unknown":
            return UNKNONW_ANSWERS
        else:
            raise ValueError("answer should be 'yes', 'no' or 'unknown'")

    def get_reponse_tempalte(self):
        return {
            "unknown":
            f"{REPLACE_PLACEHOLDER} if {self.d2.name} is in the {self.d0.name}",
            "yes":
            f"{REPLACE_PLACEHOLDER}, the {self.d2.name} is in the {self.d0.name}",
            "no":
            f"{REPLACE_PLACEHOLDER}, the {self.d2.name} is not in the {self.d0.name}",
        }


class ObjectInLocationWhat(ComplexTrackingRequest):
    answer: Literal["designated_object", "none", "unknown"]
    d0: Optional[Any] = None
    d1: Optional[Any] = None
    d2: Optional[Any] = None

    def get_question(self):
        return f"What is in the {self.d0.name}?"

    def get_answer(self) -> list[str]:
        if self.answer == "designated_object":
            return [self.d2.name]
        elif self.answer == "none":
            return OBJECTS_LOCATION_EVENT_NONE_ANSWERS
        elif self.answer == "unknown":
            return UNKNONW_ANSWERS
        else:
            raise ValueError(
                "answer should be 'designated_object', 'none' or 'unknown'")

    def get_reponse_tempalte(self):
        return {
            "unknown":
            f"{REPLACE_PLACEHOLDER} what is in the {self.d0.name}",
            "none":
            f"{REPLACE_PLACEHOLDER} is in the {self.d0.name}",
            "designated_object":
            f"the {REPLACE_PLACEHOLDER} is in the {self.d0.name}",
        }


class ObjectInLocationWhere(ComplexTrackingRequest):
    answer: Literal["designated_location", "unknown"]
    d0: Optional[Any] = None
    d1: Optional[Any] = None
    d2: Optional[Any] = None

    def get_question(self):
        return f"Where is the {self.d2.name}?"

    def get_answer(self) -> list[str]:
        if self.answer == "designated_location":
            return [self.d0.name]
        elif self.answer == "unknown":
            return UNKNONW_ANSWERS
        else:
            raise ValueError(
                "answer should be 'designated_location' or 'unknown'")

    def get_reponse_tempalte(self):
        return {
            "unknown":
            f"{REPLACE_PLACEHOLDER} where the {self.d2.name} is",
            "designated_location":
            f"the {self.d2.name} is in the {REPLACE_PLACEHOLDER}",
        }


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

    @model_validator(mode="after")
    def check_dimentions(self):
        # validate that len(dim0) is grater than 2
        if not len(self.dim0) > 2:
            raise ValueError("dim0 should have at least 3 locations")
        return self

    @property
    def as_tuple(self):
        return (self.dim0, self.dim1, self.dim2)


class ComplexTracking(BaseGenerator):
    model: Any
    states_qty: int
    topic: ComplexTrackingRequest
    uncertainty: list = None
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
    p_antilocation: float = 0.5  # Only applies in the creation of last state
    p_object_in_actor: Optional[float] = None
    p_nowhere_OR: Optional[float] = None
    method_p_nowhere_OR: Optional[Literal['fix', 'cap']] = None
    p_move_object_tx: float = 0.5
    location_matrix: Optional[Any] = None
    state_class: Optional[State] = None

    @model_validator(mode="after")
    def check_shape_and_model(self):
        model_tuple = self.model.as_tuple
        if len(model_tuple) != len(self.shape_str):
            raise ValueError(
                f"Length mismatch: 'model.as_tuple()' has length {len(model_tuple)} "
                f"but 'shape_str' has length {len(self.shape_str)}.")
        return self

    @model_validator(mode="after")
    def fill_p_object_in_actor(self):
        # If not defined, p_object_in_actor is set to 1/len(dim2)+1 (to consider nobody)
        if not self.p_object_in_actor:
            self.p_object_in_actor = 1 - (1 / (len(self.model.dim1) + 1))
        return self

    @model_validator(mode="after")
    def check_p_nowhere_OR(self):
        # If p_nowhere_OR is defined, method_p_nowhere_OR should be defined too
        # and vice versa
        if self.p_nowhere_OR and not self.method_p_nowhere_OR:
            raise ValueError(
                "If p_nowhere_OR is defined, method_p_nowhere_OR should be defined too"
            )
        if not self.p_nowhere_OR and self.method_p_nowhere_OR:
            raise ValueError(
                "If method_p_nowhere_OR is defined, p_nowhere_OR should be defined too"
            )
        return self

    # function to log p values after instance creation
    @model_validator(mode="after")
    def log_p_values(self):
        self.logger.debug("Probability values",
                         p_antilocation=self.p_antilocation,
                         p_object_in_actor=self.p_object_in_actor,
                         p_nowhere_OR=self.p_nowhere_OR,
                         method_p_nowhere_OR=self.method_p_nowhere_OR,
                         p_move_object_tx=self.p_move_object_tx)
        return self

    def load_ontology_from_topic(self) -> tuple[Callable, Callable]:
        # Define the mapping between answer types and loader functionsc
        loader_mapping: dict[type[ComplexTrackingRequest], Callable] = {
            ObjectInLocationPolar: self._object_in_location_polar,
            ObjectInLocationWhat: self._object_in_location_what,
            ObjectInLocationWhere: self._object_in_location_where,
        }

        deltas_mapping: dict[type[ComplexTrackingRequest], Callable] = {
            ObjectInLocationPolar: self.create_transitions_polar,
            # both what and where should just be a function that do nothing
            ObjectInLocationWhat: lambda *args: None,
            ObjectInLocationWhere: lambda *args: None,
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
        topic_type = type(self.topic)
        if topic_type not in loader_mapping:
            raise ValueError(
                f"Unsupported answer type: {topic_type.__name__}. "
                f"Should be one of {[cls.__name__ for cls in loader_mapping]}")
        # Set the uncertainty based on the answer type
        if uncertainty_mapping[topic_type]:
            self.uncertainty = uncertainty_mapping[topic_type]

        if state_mapping[topic_type]:
            self.state_class = state_mapping[topic_type]
        return loader_mapping[topic_type], deltas_mapping[topic_type]

    def create_ontology(self):
        f_ontology, f_deltas = self.load_ontology_from_topic()
        self.states = f_ontology()
        f_deltas()

    def _create_aux(self):
        """
        This function creates the mapping between the objects and the indexes
        """
        self.shape = (len(self.model.dim0), len(self.model.dim1),
                      len(self.model.dim2))
        self.dim0_obj_to_idx = {o: i for i, o in enumerate(self.model.dim0)}
        self.dim1_obj_to_idx = {o: i for i, o in enumerate(self.model.dim1)}
        self.dim2_obj_to_idx = {o: i for i, o in enumerate(self.model.dim2)}
        self.dim0_idx_to_obj = {i: o for i, o in enumerate(self.model.dim0)}
        self.dim1_idx_to_obj = {i: o for i, o in enumerate(self.model.dim1)}
        self.dim2_idx_to_obj = {i: o for i, o in enumerate(self.model.dim2)}
        return

    def choice(self):
        return np.random.uniform(0, 1) < self.p_move_object_tx

    def choice_known_location(self):
        """
        This function return a boolean value based on the probability of
        the antilocation.
        """
        return np.random.uniform(0, 1) > self.p_antilocation

    def get_random_location(self, min_loc_matrix: int, max_loc_matrix: int):
        """
        This function return a random index from the location matrix given
        the boundaries: min_loc_matrix (for known locations) and max_loc_matrix
        (for unknown locations).
        Only applies in the creation of the last state.
        """
        num_locations = self.shape[0] // 2
        if self.choice_known_location():
            i_l = random.randint(min_loc_matrix, num_locations)
        else:
            i_l = random.randint(
                num_locations,
                max_loc_matrix,  # self.location_matrix.shape[0] - 1
            )
        return i_l

    def refine_generation(self, objects, actor_choices, objects_nobody):
        """This functions inteand to create a mapping strcuture for
        objects and actors based on the topic and awer types.
        It has to limit the possible locations for the objects and actors.
        Args:
            objects (np.array): list of object asigned to actors.
            actor_choices (np.array): list of actors that have an assigned object.
            object[i] is in actor[i]
            objects_nobody (np.array): list of object owned by nobody.
        """
        num_locations = self.shape[0] // 2
        actor_in_location_fixed = {}
        obs_nb_locs = []
        actor = None
        if isinstance(self.topic, ObjectInLocationWhat):
            if self.topic.answer == "designated_object":
                min_loc_matrix = 1
                max_loc_matrix = self.location_matrix.shape[0] - 1
                if objects.size and objects[0] == 0:
                    actor = actor_choices[0]
                    objs_to_re_alloc = np.where(actor_choices == actor)[0]
                    objs_to_re_alloc = objs_to_re_alloc[1:]
                    if objs_to_re_alloc.size:
                        ####################
                        # OBJECT REASSIGNMENT
                        ####################
                        # generate the list of possibles actors to re-allocate
                        actr_indexes = np.arange(self.shape[1])
                        realloc_actr_indexes = np.delete(actr_indexes, actor)
                        (
                            realloc_objects,
                            realloc_actor_choices,
                            realloc_objects_nobody,
                        ) = self.assign_objects(objs_to_re_alloc,
                                                realloc_actr_indexes)
                        objects = np.delete(objects, objs_to_re_alloc)
                        actor_choices = np.delete(actor_choices,
                                                  objs_to_re_alloc)
                        # update the objects, actor_choices and objects_nobody
                        objects = np.concatenate([objects, realloc_objects])
                        actor_choices = np.concatenate(
                            [actor_choices, realloc_actor_choices])
                        objects_nobody = np.concatenate(
                            [objects_nobody, realloc_objects_nobody])
                    ####################
                    # END OBJECT REASSIGNMENT
                    ####################

                for a_i in np.unique(actor_choices):
                    i_l = (0 if actor == a_i else self.get_random_location(
                        min_loc_matrix, max_loc_matrix))
                    actor_in_location_fixed[a_i] = i_l

                for obj in objects_nobody:
                    i_l = (0 if obj == 0 else self.get_random_location(
                        min_loc_matrix, max_loc_matrix))
                    obs_nb_locs.append(i_l)

            elif self.topic.answer == "none":
                min_loc_matrix = 1
                # Here we do not allow the full nowhere case
                max_loc_matrix = self.location_matrix.shape[0] - 1 - 1
                for a_i in np.unique(actor_choices):
                    i_l = self.get_random_location(min_loc_matrix,
                                                   max_loc_matrix)
                    actor_in_location_fixed[a_i] = i_l

                for obj in objects_nobody:
                    i_l = self.get_random_location(min_loc_matrix,
                                                   max_loc_matrix)
                    obs_nb_locs.append(i_l)

            elif self.topic.answer == "unknown":
                min_loc_matrix = 1
                max_loc_matrix = self.location_matrix.shape[0] - 1
                for a_i in np.unique(actor_choices):
                    i_l = self.get_random_location(min_loc_matrix,
                                                   max_loc_matrix)
                    actor_in_location_fixed[a_i] = i_l

                for obj in objects_nobody:
                    i_l = self.get_random_location(min_loc_matrix,
                                                   max_loc_matrix)
                    obs_nb_locs.append(i_l)

            else:
                raise ValueError(
                    "Error in refine_generation, answer should be 'designated_object', 'none' or 'unknown'"  # noqa: E501
                )

        if isinstance(self.topic, ObjectInLocationWhere):
            if self.topic.answer == "designated_location":
                min_loc_matrix = 0
                max_loc_matrix = self.location_matrix.shape[0] - 1
                if objects.size and objects[0] == 0:
                    actor = actor_choices[0]
                for a_i in np.unique(actor_choices):
                    i_l = (0 if actor == a_i else self.get_random_location(
                        min_loc_matrix, max_loc_matrix))
                    actor_in_location_fixed[a_i] = i_l
                for obj in objects_nobody:
                    i_l = (0 if obj == 0 else self.get_random_location(
                        min_loc_matrix, max_loc_matrix))
                    obs_nb_locs.append(i_l)

            elif self.topic.answer == "unknown":
                min_loc_matrix = 0
                max_loc_matrix = self.location_matrix.shape[0] - 1
                if objects.size and objects[0] == 0:
                    actor = actor_choices[0]
                for a_i in np.unique(actor_choices):
                    if actor == a_i:
                        i_l = random.randint(num_locations, max_loc_matrix)
                    else:
                        i_l = self.get_random_location(min_loc_matrix,
                                                       max_loc_matrix)
                    actor_in_location_fixed[a_i] = i_l

                for obj in objects_nobody:
                    if obj == 0:
                        i_l = random.randint(num_locations, max_loc_matrix)
                    else:
                        i_l = self.get_random_location(min_loc_matrix,
                                                       max_loc_matrix)
                    obs_nb_locs.append(i_l)
            else:
                raise ValueError(
                    "Error in refine_generation, answer should be 'designated_location' or 'unknown'"  # noqa: E501
                )

        # Converting objects_nobody to a list of integers
        objects_nobody = [int(x) for x in objects_nobody]
        # Converting keys in actor_in_location_fixed to python int
        actor_in_location_fixed = {
            int(k): v
            for k, v in actor_in_location_fixed.items()
        }
        return (
            objects,
            actor_choices,
            objects_nobody,
            actor_in_location_fixed,
            obs_nb_locs,
        )

    def assign_objects(self, objects, actors):
        """
        A function that given a list of objects, and a list of actors
        (where in the actors[-1] is referenced the nobody actor), it
        returns a list of objects, a list of actors, and a list of objects
        that are assigned to the nobody actor.
        The ocurrence of an object beeing in the nobody is given by a
        binomial distribution, where the probability of an object beeing
        in the nobody is given by `self.p_object_in_actor`.
        All arrays are just index.
        Args:
            objects (np.array): list of objects to be assigned to actors.
            actors (np.array): list of actors to be assigned to objects.
        Returns:
            objects (np.array): list of objects to be assigned to actors.
            actor_choices (np.array): list of actors to be assigned to objects.
            objects_nobody (np.array): list of objects that are assigned to nobody.
        """
        num_objects = len(objects)
        nb_actor = actors[-1]
        assert nb_actor == self.shape[1] - 1, (
            "The last actor should be the nobody actor")
        actor_choices = np.random.choice(actors[:-1],
                                         num_objects,
                                         replace=True)
        actor_choices = actor_choices + 1
        nb_vector = np.random.binomial(1, self.p_object_in_actor, num_objects)
        actor_choices = nb_vector * actor_choices
        actor_choices = actor_choices - 1
        objects_nobody = objects[np.where(actor_choices < 0)[0]].tolist()
        objs_in_actors = np.where(actor_choices >= 0)[0]
        actor_choices = actor_choices[objs_in_actors]
        objects = objects[objs_in_actors]
        return objects, actor_choices, objects_nobody

    def create_transition_map(self, ):
        """
        Given a matrix with possible actor locations, this function return a dict with:
        - keys: the origen of a transition
        - values: a list of possible destinations.
        """
        n_l = self.location_matrix.shape[1] // 2
        location_with_allowed_actor_transitions = [
            np.where(row)[0].tolist() for row in self.location_matrix[:n_l * 2]
        ]
        first_half = location_with_allowed_actor_transitions[:n_l]
        second_half = location_with_allowed_actor_transitions[n_l:]

        # for the first half
        map = {}
        for i, f_i in enumerate(first_half):
            # all first half except the current one
            map[tuple(f_i)] = [i for i in first_half if i != f_i]
            # add also the element i from the second half
            map[tuple(f_i)].extend([second_half[i]])
        dict_second_half = {}
        for i_l, s_i in enumerate(second_half):
            # only the corresponding in the first half
            dict_second_half[tuple(s_i)] = [first_half[i_l]]
        # create a variable with the joined dictionaries
        map.update(dict_second_half)
        return map

    def create_transitions_polar(self):

        def process_delta(diff):
            """
            This function generate the delta in the form of:
            [(entity, coord), [origin], [end]].
            ONLY FOR THE COMPLEXTRACKING POLAR QUESTION!
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
                delta_j = [(2, 1), diff.coords.T[o][1:].tolist(),
                           diff.coords.T[e][1:].tolist()]
                return delta_j
            elif q_locations == 2:
                # due to i dont care abount the items, just pick the place of -1 and 1
                pair = diff.data
                e = np.where(pair == 1)[0][0]
                o = np.where(pair == -1)[0][0]
                delta_j = [(1, 0), diff.coords.T[o][:-1].tolist(),
                           diff.coords.T[e][:-1].tolist()]
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
            self.logger.debug("Transition", i=i, transition=transition_info)
            deltas.append(transition_info)
        self.deltas = deltas
        return

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

        self.logger.debug(
            "initialize_state:",
            i=i,
            answer=self.topic.answer,
        )
        s = self.create_random_state(i)
        t = 0
        while not condition(s.am):
            s = self.create_random_state(i)
            t += 1

        s.logger.debug("State initialized",
                      state=s,
                      answer=self.topic.answer,
                      i=i)
        return s

    def initialize_state_with_antilocations(self, i: int,
                                            condition: Callable) -> State:
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

        self.logger.debug(
            "initialize_state:",
            i=i,
            answer=self.topic.answer,
        )
        s = self.create_random_state_with_antilocations(i)
        t = 0
        while not condition(s.am):
            s = self.create_random_state_with_antilocations(i)
            t += 1

        s.actor_locations_map = s.get_actor_locations()
        s.objects_map = s.get_objects_map()
        s.logger.debug("State initialized",
                      state=s,
                      answer=self.topic.answer,
                      i=i)
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
        actors = np.arange(self.shape[1])
        # Step 2: Pick N actors (can be repeated)
        objects, actor_choices, objects_nobody = self.assign_objects(
            objects,
            actors,  # include the nobody actor
        )
        unique_actors = np.unique(actor_choices)
        location_choices = np.random.choice(self.shape[0],
                                            len(unique_actors),
                                            replace=True)
        # Create a mapping of actor -> location to ensure one location per actor
        actor_to_location = dict(zip(unique_actors, location_choices))

        sparse_matrix = DOK(shape=self.shape, dtype=np.int8, fill_value=0)
        for obj, actor in zip(objects, actor_choices):
            loc = actor_to_location[
                actor]  # Ensure the actor gets a unique location
            sparse_matrix[loc, actor, obj] = 1

        # Step 4: Add the objects in nobody to some location
        if objects_nobody:
            for obj in objects_nobody:
                loc = np.random.choice(self.shape[0])
                sparse_matrix[loc, -1, obj] = 1

        # Get actors with nothing
        AWN = np.where((sparse_matrix[:, :, :-1] == 0).to_coo().sum(
            axis=(0, 2)).todense() == sparse_matrix.shape[0] *
                       (sparse_matrix.shape[2] - 1))[0]
        AWN = np.delete(
            AWN,
            np.where(AWN == sparse_matrix.shape[1] -  # noqa: SIM300
                     1))
        # no for each actor with nothing, pick a random place, and give then the
        # `nothing`(-1) object
        if AWN.size:
            for a in AWN:
                loc = np.random.choice(sparse_matrix.shape[0])
                sparse_matrix[loc, a, -1] = 1
        s = self.state_class(am=sparse_matrix,
                             index=i,
                             verbosity=self.verbosity,
                             log_file=self.log_file)
        return s

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
        actors = np.arange(self.shape[1])
        # Step 2: Pick N actors (can be repeated)
        objects, actor_choices, objects_nobody = self.assign_objects(
            objects,
            actors,  # include the nobody actor
        )
        ############################
        # REFINE GENERATION!
        ############################
        (
            objects,
            actor_choices,
            objects_nobody,
            actor_in_location_fixed,
            obs_nb_locs,
        ) = self.refine_generation(objects, actor_choices, objects_nobody)
        self.logger.debug(
            "Refine generation",
            objects=objects,
            actor_choices=actor_choices,
            objects_nobody=objects_nobody,
            actor_in_location_fixed=actor_in_location_fixed,
            obs_nb_locs=obs_nb_locs,
        )
        ############################
        # END REFINE GENERATION!
        ############################
        unique_actors = np.unique(actor_choices)
        unique_actors = unique_actors.tolist()
        # Step 3: Assign locations for the chosen actors
        location_choices = []
        for a in unique_actors:
            i_l = actor_in_location_fixed[a]
            vector = self.location_matrix[i_l]
            idx = np.where(vector == 1)[0].tolist()
            self.logger.debug("Step 3: Assign locations for the chosen actors",
                              a=a,
                              i_l=i_l,
                              idx=idx)
            location_choices.append(idx)

        # Create a mapping of actor -> location to ensure one location per actor
        actor_to_location = dict(zip(unique_actors, location_choices))

        sparse_matrix = DOK(shape=self.shape, dtype=np.int8, fill_value=0)
        for obj, actor in zip(objects, actor_choices):
            # get the location for the actor
            # list_loc = actor_to_location[actor]
            list_loc = np.atleast_1d(actor_to_location[actor])
            for loc in list_loc:
                sparse_matrix[loc, actor, obj] = 1

        # Step 4: Add the objects in nobody to some location
        nb_location_choices = []
        for i_l in obs_nb_locs:
            vector = self.location_matrix[i_l]
            idx = np.where(vector == 1)[0].tolist()
            nb_location_choices.append(idx)
        # create a mapping of objects in nobody location
        obj_to_location = dict(zip(objects_nobody, nb_location_choices))
        self.logger.debug("Objects in nobody location",
                          obj_to_location=obj_to_location)
        for obj, loc in obj_to_location.items():
            for l_i in loc:
                sparse_matrix[l_i, -1, obj] = 1

        # Get actors with none object (a.k.a `nothing`)
        AWN = np.where((sparse_matrix[:, :, :-1] == 0).to_coo().sum(
            axis=(0, 2)).todense() == sparse_matrix.shape[0] *
                       (sparse_matrix.shape[2] - 1))[0]
        # delete nobody sparse_matrix.shape[1] - 1
        AWN = np.delete(
            AWN,
            np.where(AWN == (  # noqa: SIM300
                sparse_matrix.shape[1] - 1)))
        # no for each actor with nothing, pick a random place, and give then the
        # `nothing`(-1) object
        if AWN.size:
            min_loc_matrix = 0
            max_loc_matrix = self.location_matrix.shape[0] - 1
            for a in AWN:
                i_l = self.get_random_location(min_loc_matrix, max_loc_matrix)
                vector = self.location_matrix[i_l]
                list_loc = np.where(vector == 1)[0].tolist()
                self.logger.debug("Actor with nothing location",
                                  a=a,
                                  l=list_loc)
                for loc in list_loc:
                    sparse_matrix[loc, a, -1] = 1
        s = self.state_class(am=sparse_matrix,
                             index=i,
                             p_nowhere_OR=self.p_nowhere_OR,
                             method_p_nowhere_OR=self.method_p_nowhere_OR,
                             verbosity=self.verbosity,
                             log_file=self.log_file)
        return s

    def create_state_with_maps(self, actor_locations_map: dict,
                               objects_map: dict):
        """
        Create a state with the given actor locations and objects map.
        Args:
        actor_locations_map (dict): A mapping of actors to a list of location indices.
        objects_map (dict): A mapping containing:
        Returns:
            State: A state with the given actor and object placements represented
                as a sparse DOK matrix.
        """
        sparse_matrix = DOK(shape=self.shape, dtype=np.int8, fill_value=0)

        # Process actors and assign their objects based on actor_locations_map and
        # 'actor_object' from objects_map.
        for actor, locations in actor_locations_map.items():
            # Get the list of objects for the actor.
            # If none exist, then assign "nothing"
            actor_objects = objects_map.get("actor_object", {}).get(actor, [])
            if not actor_objects:
                # Mark as "nothing" (i.e. using the last column with index -1)
                for loc in locations:
                    sparse_matrix[loc, actor, -1] = 1
            else:
                for loc in locations:
                    for obj in actor_objects:
                        sparse_matrix[loc, actor, obj] = 1

        # Process objects_map for objects owned by nobody.
        for obj, locations in objects_map.get("object_location", {}).items():
            for loc in locations:
                # Use the `nobody` actor index
                sparse_matrix[loc, -1, obj] = 1

        return sparse_matrix

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
        if isinstance(self.topic, ObjectInLocationPolar):
            new_am = state.create_transition(self.num_transitions, condition,
                                             axis, filter)
            if new_am is None:
                self.logger.debug(
                    "Fail both: make_actor_transition & make_object_transition"
                )
                raise ValueError(
                    "None compatible solution founded for axis / transition")
            new_state = self.state_class(am=new_am,
                                         index=j,
                                         verbosity=self.verbosity,
                                         log_file=self.log_file)
        elif isinstance(self.topic,
                        (ObjectInLocationWhat, ObjectInLocationWhere)):
            # Particularly, for what and where,
            # the deltas are return directly from the create_transition
            # Hence, the f_deltas is an dummy function.
            new_state, delta = state.create_transition(
                self.num_transitions,
                self.location_to_locations_map,
                condition,
                axis,
                filter,
            )
            self.deltas[j] = delta
        return new_state

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
        self.logger.debug(
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
                states[j] = self.create_new_state(j, states[j + 1], condition,
                                                  axis)

        elif self.topic.answer == "no":
            if random.choice([0, 1]):
                # case for d2 not in d1
                i = self.states_qty - 1
                condition = lambda x: sum(x[0, :, 0]) == 0 and sum(x[-1, :, 0]
                                                                   ) == 0
                states[i] = self.initialize_state(i, condition)
                for j in list(reversed(range(i))):
                    condition = lambda x: True
                    axis = 2 if self.choice() else 1
                    states[j] = self.create_new_state(j, states[j + 1],
                                                      condition, axis)
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
                    states[j] = self.create_new_state(j, states[j + 1],
                                                      condition, axis)
                # Forward
                # This mean, that the d2 remains in d0=uncertainty
                for j in range(i + 1, len(states)):
                    condition = lambda x: sum(x[-1, :, 0]) == 1
                    axis = 2 if self.choice() else 1
                    states[j] = self.create_new_state(j, states[j - 1],
                                                      condition, axis)

        elif self.topic.answer == "unknown":
            if random.choice([0, 1]):
                # case where d3 always remain in nowhere (d3=-1)
                i = 0
                condition = lambda x: sum(x[-1, :, 0]) == 1
                self.logger.debug("Creating polar unknown, with 0",
                                  answer=self.topic.answer,
                                  i=i)
                states[i] = self.initialize_state(i, condition)
                for j in range(1, self.states_qty):
                    # remain always in nowhere
                    condition = lambda x: sum(x[-1, :, 0]) == 1
                    # chose between move and object, or move an actor
                    axis = 2 if self.choice() else 1
                    states[j] = self.create_new_state(j, states[j - 1],
                                                      condition, axis)
            else:
                # case where d3 was not in d1, and neither in nowhere in certain point.
                # (this mean it was in some dim1 != d1)
                # and it can't be in nobody.
                i = random.randint(0, self.states_qty - 2)
                self.logger.debug("Creating polar unknown, with 0",
                                  answer=self.topic.answer,
                                  i=i)
                #
                condition = (
                    lambda x: sum(x[0, :, 0]) == 0 and sum(x[
                        -1, :, 0]) == 0 and x[:, :-1, 0].to_coo().sum() == 1
                )  # This is an extra, so someone has to be in the same place
                states[i] = self.initialize_state(i, condition)
                self.logger.debug("Begin of backward")
                for j in list(reversed(range(i))):
                    # if j is the first iteration then,
                    # define a `filter` lambda function
                    condition = lambda x: True
                    axis = 2 if self.choice() else 1
                    states[j] = self.create_new_state(j, states[j + 1],
                                                      condition, axis)
                # create the states after i, that where d2 remains in nowhere (d0==-1)
                self.logger.debug("Begin of forward")
                for j in range(i + 1, len(states)):
                    condition = lambda x: sum(x[-1, :, 0]) == 1
                    # This sould force to pick the desired object to move
                    # to `nowhere` in the first iteration.
                    if j == i + 1:  # noqa: SIM108
                        filter = lambda x: x[:, :, [0]] == 1
                    else:
                        filter = None
                    axis = 2 if self.choice() else 1
                    states[j] = self.create_new_state(j, states[j - 1],
                                                      condition, axis, filter)
        else:
            raise ValueError("Invalid answer value, should be 'yes' no 'no'")

        return states

    def _object_in_location_what(self):
        d0 = self.model.dim0[0]
        d1 = self.model.dim1[0]
        d2 = self.model.dim2[0]
        self.topic.d0 = d0
        self.topic.d1 = d1
        self.topic.d2 = d2
        # for dim0, due to anti locations, i need to add each element again
        # to the list, BUT, adding the 'anti-' prefix in each element name
        anti_locations = [
            Coordenate(name=f"anti-{d.name}") for d in self.model.dim0
        ]
        self.model.dim0.extend(anti_locations)
        self.model.dim1.append(self.uncertainty[1])
        self.model.dim2.append(self.uncertainty[2])
        self._create_aux()
        self.location_matrix = operators.generate_location_matrix(
            self.shape[0] // 2)
        self.location_to_locations_map = self.create_transition_map()
        self.logger.debug(
            "Creating _object_in_location_what",
            topic=type(self.topic).__name__,
            answer=self.topic.answer,
            l=d0.name,
            a=d1.name,
            o=d2.name,
            location_matrix_MB="{:.2f} MB".format(self.location_matrix.nbytes /
                                                  1024 / 1024),
        )

        states = [None] * self.states_qty
        # In _object_in_location_what
        # deltas are computed in create_new_state
        self.deltas = [None] * (self.states_qty - 1)
        i = self.states_qty - 1
        # What-Questions: A question of the form “What is in l?” or similar.
        if self.topic.answer == "designated_object":

            def check_designated_object(x) -> bool:
                n_l = x.shape[0] // 2
                in_designated_location = x[0, :, 0].to_coo().sum() == 1
                y = int(list(x[0, :, 0].data.keys())[0][0])
                f_half = x[:, y, 0].todense()[:n_l]
                s_half = x[:, y, 0].todense()[n_l:]
                # validate that vector of ones is generated.
                valid_designated_ok = list(f_half + s_half) == ([1] * n_l)
                # For the corresponding anti-location, for all the other objects (1:-1)
                # there should be certainty that they are in the anti-location.
                current_in_not_l = np.array(
                    x[n_l, :, 1:-1].to_coo().sum(axis=0).todense())
                ideal_not_in_l = np.array([1] * (x.shape[2] - 2))
                answer_others_not_in_l = np.array_equal(
                    current_in_not_l, ideal_not_in_l)
                # None of the others objects (1:-1) can be in the full-nowhere location
                # Thas's why (n_l:).
                not_object_in_full_nowhere = np.all(
                    x[n_l:, :, 1:-1].to_coo().sum(axis=0) < n_l)
                return (in_designated_location and valid_designated_ok
                        and answer_others_not_in_l
                        and not_object_in_full_nowhere)

            condition = check_designated_object
        elif self.topic.answer == "none":

            def check_none(x) -> bool:
                n_l = x.shape[0] // 2
                is_empty_location = x[0, :, :-1].to_coo().sum() == 0
                # For the anti-location designated, all Os added by column, are there.
                current_not_in_l = np.array(
                    x[n_l, :, :-1].to_coo().sum(axis=0).todense())
                ideal_not_in_l = np.array([1] * (x.shape[2] - 1))
                all_not_in_l = np.array_equal(current_not_in_l, ideal_not_in_l)
                # None objects in the full-nowhere location
                not_object_in_full_nowhere = np.all(
                    x[n_l:, :, :-1].to_coo().sum(axis=0) < n_l)
                return is_empty_location and all_not_in_l and not_object_in_full_nowhere

            condition = check_none
        elif self.topic.answer == "unknown":

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

        else:
            raise ValueError(
                "Invalid answer should be 'designated_object', 'none' or 'unknown'"
            )

        states[i] = self.initialize_state_with_antilocations(i, condition)
        for j in list(reversed(range(i))):
            condition = lambda x: True
            axis = 2 if self.choice() else 1
            states[j] = self.create_new_state(j, states[j + 1], condition,
                                              axis)

        groups, actor_locations_map, objects_map, i, e = gr._get_forward(
            states=states,
            deltas=self.deltas,
            n_locs=self.shape[0],
            nobody=self.shape[1] - 1,
            logger=self.logger,
        )

        if isinstance(e, Exception):
            self.logger.debug("FORWARD PASS",
                              error=e,
                              groups=groups,
                              actor_locations_map=actor_locations_map,
                              objects_map=objects_map,
                              i=i)
            raise e
        else:
            am = self.create_state_with_maps(actor_locations_map, objects_map)
            if condition(am):
                self.logger.debug("Forward pass OK")
            else:
                self.logger.debug("Forward pass failed")
                raise ValueError("Forward pass failed")

        return states

    def _object_in_location_where(self):
        d0 = self.model.dim0[0]
        d1 = self.model.dim1[0]
        d2 = self.model.dim2[0]
        self.topic.d0 = d0
        self.topic.d1 = d1
        self.topic.d2 = d2
        # for dim0, due to anti locations, i need to add each element again
        # to the list, BUT, adding the 'anti-' prefix in each element name
        anti_locations = [
            Coordenate(name=f"anti-{d.name}") for d in self.model.dim0
        ]
        self.model.dim0.extend(anti_locations)
        self.model.dim1.append(self.uncertainty[1])
        self.model.dim2.append(self.uncertainty[2])
        self._create_aux()
        self.location_matrix = operators.generate_location_matrix(
            self.shape[0] // 2)
        self.location_to_locations_map = self.create_transition_map()
        self.logger.debug(
            "Creating _object_in_location_where",
            topic=type(self.topic).__name__,
            answer=self.topic.answer,
            l=d0.name,
            a=d1.name,
            o=d2.name,
            location_matrix_MB="{:.2f} MB".format(self.location_matrix.nbytes /
                                                  1024 / 1024),
        )

        states = [None] * self.states_qty
        self.deltas = [None] * (self.states_qty - 1)
        i = self.states_qty - 1
        if self.topic.answer == "designated_location":

            def check_designated_location(x) -> bool:
                n_l = x.shape[0] // 2
                in_designated_location = x[0, :, 0].to_coo().sum() == 1
                y = int(list(x[0, :, 0].data.keys())[0][0])
                f_half = x[:, :, 0].to_coo().T[y][:n_l]
                s_half = x[:, :, 0].to_coo().T[y][n_l:]
                # validate that vector of ones is generated.
                valid_designated_ok = list(
                    (f_half + s_half).todense()) == ([1] * n_l)
                return in_designated_location and valid_designated_ok

            condition = check_designated_location
        elif self.topic.answer == "unknown":

            def check_unknown(x) -> bool:
                n_l = x.shape[0] // 2
                not_in_known_location = x[:n_l, :, 0].to_coo().sum() == 0
                # TODO: Check that is in valids anti-locations
                return not_in_known_location

            condition = check_unknown
        else:
            raise ValueError(
                "Invalid answer value, should be 'designated_location' or 'unknown'"
            )

        states[i] = self.initialize_state_with_antilocations(i, condition)
        for j in list(reversed(range(i))):
            condition = lambda x: True
            axis = 2 if self.choice() else 1
            states[j] = self.create_new_state(j, states[j + 1], condition,
                                              axis)

        # TODO: It will be required a forward pass.
        # This forward pass will need to catch the deltas, and based on that update all
        # the state information.
        # The implications of this will be defined later.

        return states

    def create_fol(self):

        def enumerate_model(element: list, shape_type: str) -> list[list]:
            enumeration = []
            for e in element:
                if e != self.uncertainty:
                    enumeration.append(Exists(thing=e, shape_str=shape_type))
            return enumeration

        def describe_states(state) -> list[list]:
            state_sentences = []
            if isinstance(self.topic, ObjectInLocationPolar):
                # 1) Actors with nothing in a place.
                actors_in_locations_nothing = state.am[:-1, :-1, -1]
                for loc, a in actors_in_locations_nothing.data:
                    state_sentences.append(
                        In(
                            entity=self.dim1_idx_to_obj[a],
                            coordenate=self.dim0_idx_to_obj[loc],
                            shape_str=(self.shape_str[0], self.shape_str[1]),
                        ))
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
                                    shape_str=(self.shape_str[0],
                                               self.shape_str[1]),
                                ))
                    state_sentences.append(
                        In(
                            entity=self.dim2_idx_to_obj[o],
                            coordenate=self.dim1_idx_to_obj[a],
                            shape_str=(self.shape_str[1], self.shape_str[2]),
                        ))

                # 3) Objects in a place
                objects_in_locations = state.am[:-1, -1, :-1]
                for loc, o in objects_in_locations.data:
                    state_sentences.append(
                        In(
                            entity=self.dim2_idx_to_obj[o],
                            coordenate=self.dim0_idx_to_obj[loc],
                            shape_str=(self.shape_str[0], self.shape_str[2]),
                        ))
                # Regarding actos in nowhere, there is no sentece.
            elif isinstance(self.topic,
                            (ObjectInLocationWhat, ObjectInLocationWhere)):

                def aux_FOL_creation(entity, loc, absolute_nowhere, dict_e,
                                     dict_c, shape_str):
                    """
                    A function that given an entity (object or actor), and a location,
                    it returns a list of sentences FOL sentences.
                    Args:
                        entity (int): The index of the entity.
                        loc (list): Indexes that represent the location of the entity.
                        absolute_nowhere (list): Absolute nowhere location indexes.
                        dict_e (dict): Mapping dict from entity index to class.
                        dict_c (dict): Mapping dict from location index to class.
                        shape_str (tuple): Type of FOL description.
                    Returns:
                        sentences: List of FOL.
                    """
                    sentences = []
                    anti_loc = absolute_nowhere[0]
                    if loc != absolute_nowhere:
                        if loc[0] < absolute_nowhere[0]:
                            # this is known known location!
                            loc = loc[0]
                            sentences.append(
                                In(
                                    entity=dict_e[entity],
                                    coordenate=dict_c[loc],
                                    shape_str=shape_str,
                                ))
                        else:
                            # This is correspond to Out locations!
                            for l_i in loc:
                                sentences.append(
                                    Out(
                                        entity=dict_e[entity],
                                        coordenate=dict_c[l_i - anti_loc],
                                        shape_str=shape_str,
                                    ))
                    return sentences

                absolute_nowhere = list(
                    range(self.shape[0] // 2, self.shape[0]))
                nothing_object = self.shape[2] - 1
                # Actors
                for a, loc in state.actor_locations_map.items():
                    state_sentences.extend(
                        aux_FOL_creation(
                            a, loc, absolute_nowhere, self.dim1_idx_to_obj,
                            self.dim0_idx_to_obj,
                            (self.shape_str[0], self.shape_str[1])))
                    # Objects with owner
                    if a in state.objects_map['actor_object']:
                        for o in state.objects_map['actor_object'][a]:
                            if o != nothing_object:
                                state_sentences.append(
                                    In(
                                        entity=self.dim2_idx_to_obj[o],
                                        coordenate=self.dim1_idx_to_obj[a],
                                        shape_str=(self.shape_str[1],
                                                   self.shape_str[2]),
                                    ))
                # Objects without owner
                for o, loc in state.objects_map['object_location'].items():
                    loc = list(loc)
                    state_sentences.extend(
                        aux_FOL_creation(
                            o, loc, absolute_nowhere, self.dim2_idx_to_obj,
                            self.dim0_idx_to_obj,
                            (self.shape_str[0], self.shape_str[2])))

            return state_sentences

        def describe_transitions(state: State) -> list[list]:
            i = state.index
            delta = self.deltas[i]

            # Define mapping for different delta cases
            delta_mappings = {
                (1, 0): (
                    self.dim1_idx_to_obj,
                    self.dim0_idx_to_obj,
                    ("locations", "actors"),
                    self.uncertainty[0],
                ),
                (2, 1): (
                    self.dim2_idx_to_obj,
                    self.dim1_idx_to_obj,
                    ("actors", "objects"),
                    self.uncertainty[1],
                ),
            }

            if delta[0] not in delta_mappings:
                raise ValueError("Invalid delta")
            entity_map, coord_map, shape_str, uncertainty = delta_mappings[
                delta[0]]
            idx_entity = delta[1][1]
            idx_prev_coord = delta[1][0]
            idx_next_coord = delta[2][0]

            # Common logic for Polar and delta[0] == (2, 1) cases
            if isinstance(self.topic, ObjectInLocationPolar) or (
                    isinstance(self.topic,
                               (ObjectInLocationWhat, ObjectInLocationWhere))
                    and delta[0] == (2, 1)):
                entity = entity_map[idx_entity]
                prev_coord = coord_map[idx_prev_coord]
                next_coord = coord_map[idx_next_coord]

                if prev_coord == uncertainty:
                    transition_sentences = To(entity=entity,
                                              coordenate=next_coord,
                                              shape_str=shape_str)
                elif next_coord == uncertainty:
                    transition_sentences = From(entity=entity,
                                                coordenate=prev_coord,
                                                shape_str=shape_str)
                else:
                    transition_sentences = random.choice([
                        To(entity=entity,
                           coordenate=next_coord,
                           shape_str=shape_str),
                        FromTo(
                            entity=entity,
                            coordenate1=prev_coord,
                            coordenate2=next_coord,
                            shape_str=shape_str,
                        ),
                    ])
            elif isinstance(self.topic,
                            (ObjectInLocationWhat,
                             ObjectInLocationWhere)) and delta[0] == (1, 0):
                # In Tx Actor its necesary to handle the location vector
                # that can contain anti-locations.
                entity = entity_map[idx_entity]
                # get the index where start antilocations
                anti_loc = self.shape[0] // 2
                # check if the prev_coord or next_coord are in any antilocation
                if idx_prev_coord[0] >= anti_loc:
                    next_coord = coord_map[idx_next_coord[0]]
                    transition_sentences = To(entity=entity,
                                              coordenate=next_coord,
                                              shape_str=shape_str)
                elif idx_next_coord[0] >= anti_loc:
                    prev_coord = coord_map[idx_prev_coord[0]]
                    transition_sentences = From(entity=entity,
                                                coordenate=prev_coord,
                                                shape_str=shape_str)
                else:
                    prev_coord = coord_map[idx_prev_coord[0]]
                    next_coord = coord_map[idx_next_coord[0]]
                    transition_sentences = random.choice([
                        To(entity=entity,
                           coordenate=next_coord,
                           shape_str=shape_str),
                        FromTo(
                            entity=entity,
                            coordenate1=prev_coord,
                            coordenate2=next_coord,
                            shape_str=shape_str,
                        ),
                    ])

            return [transition_sentences]

        world_enumerate = []
        story = []

        for t, dim_str in zip(self.model.as_tuple, self.shape_str):
            if isinstance(self.topic,
                          (ObjectInLocationWhat,
                           ObjectInLocationWhere)) and dim_str == "locations":
                world_enumerate.extend(
                    enumerate_model(t[:self.shape[0] // 2], dim_str))
            else:
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
            response_templates=self.topic.get_reponse_tempalte(),
        )

        self.fol = world_enumerate + story

    def create_nl(self):
        self.nl = [f.to_nl() for f in self.fol]

    def generate(self):
        self.create_ontology()
        self.create_fol()

    def get_json(self):
        json = self.story.create_json()
        options = list(get_type_hints(self.topic)['answer'].__args__)
        contextualized_options = dict()
        if isinstance(self.topic, ObjectInLocationPolar):
            contextualized_options["yes"] = ["yes"]
            contextualized_options["no"] = ["no"]
            contextualized_options["unknown"] = [UNKNONW_ANSWERS[0]]
        elif isinstance(self.topic, ObjectInLocationWhat):
            options.remove('designated_object')
            aux = [o.name for o in self.model.dim2]
            options.extend(aux)
            contextualized_options["designated_object"] = list()
            for o in aux:
                # the designated_object list contains the none-answers,
                # which have a different context template.
                if o not in OBJECTS_LOCATION_EVENT_NONE_ANSWERS:
                    contextualized_options["designated_object"].append(o)

            options.remove('none')
            contextualized_options["none"] = [
                OBJECTS_LOCATION_EVENT_NONE_ANSWERS[0]
            ]

            # add unknown case
            contextualized_options["unknown"] = [UNKNONW_ANSWERS[0]]

        elif isinstance(self.topic, ObjectInLocationWhere):
            options.remove('designated_location')
            aux = [loc.name for loc in self.model.dim0[:self.shape[0] // 2]]
            options.extend(aux)
            contextualized_options["designated_location"] = aux

            # add unknown case
            contextualized_options["unknown"] = [UNKNONW_ANSWERS[0]]

        random.shuffle(options)
        json["options"] = options

        # Add contextualized responses
        json["contextualized_options"] = list()
        for key in contextualized_options:
            random.shuffle(contextualized_options[key])
            for element in contextualized_options[key]:
                json["contextualized_options"].append(
                    self.story.response_templates[key].replace(
                        REPLACE_PLACEHOLDER, element))
        json["contextualized_answer"] = list()
        for element in self.story.answer:
            json["contextualized_answer"].append(
                self.story.response_templates[self.topic.answer].replace(
                    REPLACE_PLACEHOLDER, element))

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
