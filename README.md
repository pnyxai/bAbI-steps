# Title: TODO

# Abstract: TODO

# Topics: TODO

# Babisteps Dataset Generator

`main.py`, generates datasets of stories based on the task configurations found in the tasks folder. It performs the following operations:

- Reads task configurations from a `commons.yaml` file inside the specified tasks directory.
- Loads task modules dynamically using functions defined in the tasks.
- Generates stories using the tasks' generators.
- Saves the generated stories in JSONL and TXT formats.
- Creates a final dataset from the generated outputs.

## Command-line Arguments

The following command-line arguments can be used to customize the behavior of the script:

- **`--task_path:`** Root path where the task configuration folders are located. 
- **`--tasks:`** List of task names to run the experiments. Use a comma-separated string (e.g., `--tasks task1,task2`).
- **`--output_path:`** Path to save the script results. The output folder will be timestamped.
- **`--seed:`** Random seed for Python's random module.
- **`--numpy_random_seed:`** Random seed for NumPy's random module.
- **`--verbosity:`** or **`-v`** Controls the reported logging error level. (`CRITICAL|ERROR|WARNING|INFO|DEBUG`)

## Example Usage

Below are a few examples on how you might run the script:

### Run All Tasks with Default Settings
```sh
python3 main.py
```