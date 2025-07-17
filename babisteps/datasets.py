import os
from pathlib import Path

from datasets import load_dataset

TASKS2NAME = {
    "1": "simpletracking",
    "2": "immediateorder",
    "3": "complextracking",
    "4": "listing",
    "5": "sizeorder",
    "6": "spatialorder",
    "7": "temporalorder",
}

NAME2TASK = {value: key for key, value in TASKS2NAME.items()}

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
ds = load_dataset('PnyxAI/babisteps', '<task>')
```

The available tasks are:
| Task ID | Task Name | Split Name|
|---------|-----------|----|
| 1       | simpletracking| train, validation, test |
| 2       | immediateorder| train, validation, test |
| 3       | complextracking| train, validation, test |
| 4       | listing| train, validation, test |
| 5       | sizeorder| train, validation, test |
| 6       | spatialorder| train, validation, test |
| 7       | temporalorder| train, validation, test |


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
        - split: {split_name}
          path: "dataset/{folder_name}/{folder_name}-{split_name}.parquet"
"""
    return string


def create_babisteps_dataset(dataset_path: Path, jsonl_path_dict: dict, logger,
                             splits):
    os.mkdir(os.path.join(dataset_path, "dataset"))

    # Create readme
    with open(os.path.join(dataset_path, "README.md"), "w") as f:
        # Write header part
        f.write(header)
        config_lines = []
        for task_id, (task_name,
                      jsonl_task_path) in enumerate(jsonl_path_dict.items()):
            folder_name = f"{task_name}"
            os.mkdir(os.path.join(dataset_path, "dataset", folder_name))
            logger.info("Loading task into HF dataset",
                        task_name=task_name,
                        jsonl_task_path=jsonl_task_path)
            # with jsonl_task_path like:
            # PosixPath('outputs/2025-06-28T00:02:06/jsonl/spatialorder/test.jsonl')
            # should create data_files like:
            # {'train':'x/train.json', 'validation':'x/valid.json','test':'x/test.json'}
            string = f"""\
    - config_name: {folder_name}
      data_files:
"""
            config_lines.append(string)
            data_files = {}
            for split in splits:
                if split == "test":
                    data_files[split] = str(jsonl_task_path)
                else:
                    data_files[split] = str(jsonl_task_path.parent /
                                            f"{split}.jsonl")
            ds = load_dataset("json", data_files=data_files)
            #ds["test"] = ds.pop("train")
            for split in ds:
                ds[split].to_parquet(
                    os.path.join(dataset_path, "dataset", folder_name,
                                 f"{folder_name}-{split}.parquet"))
                config_lines.append(_craete_config_yaml(folder_name, split))

        # Write config lines
        for line in config_lines:
            f.write(line)
        # Write footer part
        f.write(footer)
        f.close()
    return
