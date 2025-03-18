* `task_name`: name of the taks
* `objects`: # of objects in the story
* `actors`: # of actors in the story
* `locations`: # of locations in the story
* `gen_kwargs`: Set of possible generation probabilities
  - `p_antilocation`: probability (P) that at the end of the story an object is in an anti-location.
  - `p_object_in_actor`: P that an object is in an actor at the end of the story.
  - `p_nowhere_OR`: When applying, P of selecting the absolute `nowhere`. this interact with `method_p_nowhere_OR`.
  - `method_p_nowhere_OR`: Method of considering the `p_nowhere_OR` probability. 
    - `cap`: takes the minimum between `p_nowhere_OR` and the original probability.
    - `fix`: fixes the probability to `p_nowhere_OR`.
  - `p_move_object_tx`: P that next transition is a movement of an object. Diminishing this probability increases the probability of actor movements.

* `func`: name of the function returning an iterable of story generators (like utils.<itarable of generators>)