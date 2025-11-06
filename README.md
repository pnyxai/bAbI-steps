# bAbI-steps Dataset Generator

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
- **`--include_lm_eval_templates:`** Flag to include the `lm-eval-harness` custom template in the output path. 
- **`--seed:`** Random seed for Python's random module.
- **`--numpy_random_seed:`** Random seed for NumPy's random module.
- **`--verbosity:`** or **`-v`** Controls the reported logging error level. (`CRITICAL|ERROR|WARNING|INFO|DEBUG`)
- **`--num_workers:`** Number of parallel workers for task processing. Set to 1 to disable parallelization.
- **`--optimistic_generation`:** If `1` (default), skip failed samples and continue. If `0`, fail the entire process when max generation retries are reached.

## Example Usage

Below are a few examples on how you might run the script:

### Run All Tasks with Default Settings
```sh
python3 main.py
```

### Evaluate with `lm-eval-harenss`

Before continue, be sure you have installed `lm-eval-harness`

##### `local-completions`

```bash
lm_eval \
--model local-completions \
--model_args model=<SERVED_MODEL_NAME>,base_url=http://{ip:port}/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,tokenizer_backend=None \
--limit 3 \
--tasks babisteps-all \
--include_path <PATH/TO/LM/EVAL/TEMPLATES>  \
--output_path <PATH/WHERE/TO/SAVE/RESULTS> \
--log_samples
```

##### `local-chat-completions`

```bash
lm_eval \
--model local-chat-completions \
--model_args model=<SERVED_MODEL_NAME>,base_url=http://{ip:port}/v1/chat/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,tokenizer_backend=None \
--gen_kwargs max_tokens=8192,timeout=600 \
--limit 3 \
--tasks babisteps-chat-cot-all \
--include_path <PATH/TO/LM/EVAL/TEMPLATES> \
--output_path <PATH/WHERE/TO/SAVE/RESULTS> \
--apply_chat_template \
--fewshot_as_multiturn \
--log_samples
```

>Note:
>
> * Reasoner models would consume much more tokens, probably you should tweak `--gen_kwargs max_tokens=8192,timeout=600` params accordingly.
>
> * Remove `--limit x` if you want to evaluate over the entire generated dataset.


# Paper-Title: TODO
Abstract:
> TODO

## Cite as:
```
TODO
```