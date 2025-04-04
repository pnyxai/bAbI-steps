

# Commons

* `task_path`: relative path where to find common.yaml file
* `max_retries`: max number of retries when generating a story
* `num_samples_by_task`: number of stories to generate for each task
* `verbosity`: strings that control the logger level:
  - `"DEBUG"`
  - `"INFO"`
  - `"WARNING"`
  - `"ERROR"`
  - `"CRITICAL"`
* `states_qty`: for `SimpleTracking` and `ComplexTracking`, the number of states that compose the story.
* `edges_qty`: for `ImmediateOrder`, the number of edges that compose the story **for each relation**. The # of relations are defined in the task folder.
* `locations`: list of locations.
* `actors`: list of actors names.
* `objects`: list of objects names.