import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from multiprocessing import Manager, Process, Queue
from pathlib import Path

import yaml

from babisteps import datasets as ds
from babisteps import logger
from babisteps import proccesing as proc
from babisteps import utils as ut

def worker_wrapper(tn, tp, cfg, lq, so, opt_gen, res, errs):
    try:
        result = proc.process_single_task(
            tn, tp, cfg, lq, so, opt_gen)
        res.append(result)
    except Exception as e:
        errs.append((tn, str(e)))

def main():
    parser = argparse.ArgumentParser(
        description="Create a dataset of stories (multiprocessing version)")
    parser.add_argument(
        "--task_path",
        type=str,
        default="./babisteps/tasks",
        help="Root path where are placed the tasks",
    )
    parser.add_argument(
        "--tasks",
        type=ut.split_or_empty,
        default=["all"],
        help="List of the taks name to run the experiments",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs",
        help="Path to save the script results",
    )
    parser.add_argument(
        "--include_lm_eval_template",
        default=True,
        help="Include the lm_eval template in the output folder",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for python's random module. Default set to 0.",
    )
    parser.add_argument(
        "--numpy_random_seed",
        type=int,
        default=1234,
        help="Random seed for numpy's random module. Default set to 1234.",
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help=
        "Number of parallel workers for task processing. Set to 1 to disable parallelization.",
    )
    parser.add_argument(
        "--optimistic_generation",
        type=int,
        choices=[0, 1],
        default=1,
        help=
        ("If 1 (default), skip failed samples and continue. "
         "If 0, fail the entire process when max generation retries are reached."
         ),
    )
    args = parser.parse_args()

    # Get base output folder path
    output_path = Path(args.output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # Add date string
    current_date = datetime.now()
    output_path = Path(
        os.path.join(output_path, current_date.strftime("%Y-%m-%dT%H:%M:%S")))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    dataset_path = Path(os.path.join(output_path, "babisteps"))
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    jsonl_path = Path(os.path.join(output_path, "jsonl"))
    if not os.path.exists(jsonl_path):
        os.mkdir(jsonl_path)

    start_time = time.time()

    # Setup logging queue and process
    log_queue = Queue()
    log_file = os.path.join(output_path, "logs.txt")
    log_process = Process(target=logger.logger_process,
                          args=(log_queue, log_file, args.verbosity))
    log_process.start()

    main_logger = logger.QueueLogger(log_queue)

    task_path = Path(args.task_path)
    if not os.path.exists(task_path):
        raise ValueError(f"Path {task_path} does not exist")

    # get task_folders as the folder inside the task_path (each element as Path)
    task_folders = [f for f in task_path.iterdir() if f.is_dir()]
    common_yaml = task_path / "commons.yaml"
    if not os.path.exists(common_yaml):
        raise ValueError(f"File {common_yaml} does not exist")

    # load common.yaml
    with open(common_yaml) as stream:
        try:
            yaml_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise ValueError(f"Error loading {common_yaml}") from exc

    yaml_cfg["output_path"] = jsonl_path
    yaml_cfg["seed"] = args.seed
    yaml_cfg["numpy_random_seed"] = args.numpy_random_seed

    task_path_dict = ut.create_task_path_dict(args.tasks, task_folders,
                                              main_logger)

    seed_message = []
    if args.seed is not None:
        seed_message.append(f"Setting random seed to {args.seed}")
    if args.numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {args.numpy_random_seed}")
    if seed_message:
        main_logger.info(" | ".join(seed_message))

    try:
        # Parallel or sequential processing
        if args.num_workers > 1:
            main_logger.info("Starting parallel processing",
                             num_workers=args.num_workers)

            # Use Manager for shared results and errors
            manager = Manager()
            results = manager.list()
            errors = manager.list()

            processes = []
            task_items = list(task_path_dict.items())

            for idx, (task_name_i, task_path_i) in enumerate(task_items):
                # Wait if we've reached max workers
                while len(processes) >= args.num_workers:
                    for p in processes[:]:
                        if not p.is_alive():
                            p.join()
                            processes.remove(p)
                    time.sleep(0.1)

                # Start new process with a wrapper function to append results
                

                p = Process(target=worker_wrapper,
                            args=(task_name_i, task_path_i, yaml_cfg,
                                  log_queue, idx, args.optimistic_generation,
                                  results, errors))
                p.start()
                processes.append(p)

            # Wait for all processes to complete and check for failures
            failed_processes = []
            for p in processes:
                p.join()
                if p.exitcode != 0:
                    failed_processes.append(p)

            # If any process failed, raise an error
            if failed_processes or len(errors) > 0:
                main_logger.error("One or more worker processes failed",
                                  num_failed=len(failed_processes),
                                  num_errors=len(errors))
                if errors:
                    for task_name, error_msg in errors:
                        main_logger.error("Task failed",
                                          task=task_name,
                                          error=error_msg)
                # Give logger time to flush error messages before raising
                time.sleep(0.5)
                raise RuntimeError(
                    f"{len(failed_processes)} worker process(es) failed with non-zero exit code. "
                    f"{len(errors)} worker(s) reported errors. Check logs for details."
                )

            jsonl_path_dict = dict(results)

        else:
            # Sequential processing (original behavior)
            main_logger.info("Running in sequential mode (num_workers=1)")
            jsonl_path_dict = {}
            for idx, (task_name_i,
                      task_path_i) in enumerate(task_path_dict.items()):
                result = proc.process_single_task(task_name_i, task_path_i,
                                                  yaml_cfg, log_queue, idx,
                                                  args.optimistic_generation)
                jsonl_path_dict[result[0]] = result[1]

        main_logger.info("STARTING DATASET CREATION")
        ds.create_babisteps_dataset(dataset_path, jsonl_path_dict, main_logger,
                                    proc.SPLITS)
        main_logger.info("SUCCESS DATASET CREATION")

        # SAVE STATUS CSV
        main_logger.info("SAVING STATUS CSV")
        ut.save_status_csv(output_path, jsonl_path_dict,
                           yaml_cfg["num_samples_by_task"], main_logger)

        # LM-EVAL-HARNESS TEMPLATE CREATION
        if args.include_lm_eval_template:
            main_logger.info("STARTING LM-EVAL-HARNESS TEMPLATE CREATION")
            src_path = Path(__file__).parent / "babisteps" / "lm_eval"
            lm_eval_path = Path(os.path.join(output_path, "lm_eval"))
            if not os.path.exists(lm_eval_path):
                os.mkdir(lm_eval_path)
            if not os.path.exists(src_path):
                raise ValueError(f"Path {src_path} does not exist")

            shutil.copytree(src_path, lm_eval_path, dirs_exist_ok=True)

            df_yamls = [
                file
                for file in Path(lm_eval_path).rglob("*_default_template_yaml")
            ]
            for file in df_yamls:
                if not ut.replace_path_in_file(file, str(dataset_path),
                                               main_logger):
                    main_logger.warning("File not found", file=file)

        end_time = time.time()
        main_logger.info("Execution time", time=end_time - start_time)

    finally:
        # Always cleanup logger process, even if there's an error
        # Give logger process time to flush all messages
        time.sleep(0.5)

        # Stop logger process
        log_queue.put(None)
        log_process.join(timeout=5)  # Wait max 5 seconds for clean shutdown

        # If still alive, terminate forcefully
        if log_process.is_alive():
            log_process.terminate()
            log_process.join()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
