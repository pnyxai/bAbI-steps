import os
from pathlib import Path

from datasets import load_dataset

# TODO: Complet header and footer
header = """---
language:
    - en
license: mit
task_categories:
    - question-answering
pretty_name: bAbI-steps divided by task/skills
description: "TODO"
configs:
"""

footer = """---
# bAbI-steps Dataset : Per-Task

```python
from datasets import load_dataset
ds = load_dataset('PnyxAI/babisteps', <task>')
```

The available tasks are:
| Task ID | Task Name | Split Name|
|---------|-----------|----|
| I       | TODO| TODO |

### PAPER NAME

@article{TODO,
title={TODO},
author={TODO},
journal={TODO},
year={TODO}
}
"""


def _craete_config_yaml(folder_name, split_name):
    string = f"""\
    - config_name: {folder_name}
      data_files:
        - split: {split_name}
          path: "data/{folder_name}/{folder_name}-{split_name}.parquet"
"""
    return string


def create_babisteps_dataset(dataset_path: Path, jsonl_path_dict: dict,
                             logger):
    os.mkdir(os.path.join(dataset_path, "data"))

    # Create readme
    with open(os.path.join(dataset_path, "README.md"), "w") as f:
        # Write header part
        f.write(header)
        config_lines = []
        for task_id, (task_name,
                      jsonl_task_path) in enumerate(jsonl_path_dict.items()):
            folder_name = f"{task_name}"
            os.mkdir(os.path.join(dataset_path, "data", folder_name))
            logger.debug("Loading task into HF dataset", task_name=task_name)
            ds = load_dataset("json", data_files=os.fspath(jsonl_task_path))
            for split in ds:
                ds[split].to_parquet(
                    os.path.join(dataset_path, "data", folder_name,
                                 f"{folder_name}-{split}.parquet"))
                config_lines.append(_craete_config_yaml(folder_name, split))
        print("config_lines", config_lines)
        # Write config lines
        for line in config_lines:
            f.write(line)
        # Write footer part
        f.write(footer)
        f.close()
    return
