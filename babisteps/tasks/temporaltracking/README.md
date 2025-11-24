* `task_name`: name of the task
* `entities`: # of possible entities in the story
* `coordinates`: # of possible coordinates in the story
* `edge_qty`: # of edges (previous transitive reduction) that compose the relation matrix.
* `events_qty`: # of events in the story.
* `relations_qty`: # of relations in the story (here only 1, only `temporal`).
* `gen_kwargs`: Set of possible generation args if apply
* `func`: name of the function returning an iterable of story generators (like utils.<itarable of generators>)

>Note: The relation matrix will be of size `events_qty` x `events_qty` filled with `edge_qty` edges representing the temporal relations between events. Then, the transitive reduction will be applied to get the final relation matrix where each relation is a line of the story.