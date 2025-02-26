import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from babisteps import logger
from babisteps import proccesing as proc


def split_or_empty(value):
    return value.split(",") if value else []


def main():
    parser = argparse.ArgumentParser(description="Create a dataset of stories")
    parser.add_argument("--q_stories",
                        type=int,
                        help="Number of stories to create")
    parser.add_argument("--task",
                        choices=["simpletracking", "complextracking"])
    parser.add_argument("--states_lenght",
                        type=int,
                        help="Lenght of states by story")
    parser.add_argument("--locations",
                        help="List of locations",
                        type=split_or_empty,
                        default=[])
    parser.add_argument("--actors",
                        help="List of actors",
                        type=split_or_empty,
                        default=[])
    parser.add_argument("--objects",
                        help="List of objects",
                        type=split_or_empty,
                        default=[])
    parser.add_argument("--question", type=str, help="Question to ask")
    parser.add_argument("--answer", type=str, help="Answer to the question")
    parser.add_argument("--path",
                        type=str,
                        default="./outputs",
                        help="Path to save the script results")
    parser.add_argument(
        "--gen_kwargs",
        type=str,
        default=None,
        help=("String arguments for story generation,"
              " e.g. `p_antilocation=0.5,p_object_in_actor=0.75`"),
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level.",
    )
    args = parser.parse_args()

    # Get base output folder path
    output_path = Path(args.path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # Add date string
    current_date = datetime.now()
    output_path = Path(
        os.path.join(output_path, current_date.strftime("%Y-%m-%dT%H:%M:%S")))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # init of excecution
    start_time = time.time()
    main_logger = logger.get_logger("main",
                                    level=getattr(logging,
                                                  f"{args.verbosity}"),
                                    log_file=os.path.join(
                                        output_path, "logs.txt"))

    if args.task == "simpletracking":
        # Count how many lists are non-empty
        non_empty_count = sum(
            bool(lst) for lst in [args.locations, args.actors, args.objects])
        assert non_empty_count == 2, (
            "The number non-empty lists from locations, actors, and objects must be 2"
        )
        jsonl_dataset, txt_dataset, folder_path = proc.create_simpletracking(
            q_stories=args.q_stories,
            states_qty=args.states_lenght,
            locations=args.locations,
            actors=args.actors,
            objects=args.objects,
            question=args.question,
            answer=args.answer,
            path=output_path,
            verbosity=getattr(logging, f"{args.verbosity}"),
            logger=main_logger,
            gen_kwargs=args.gen_kwargs,
        )
    elif args.task == "complextracking":
        # Count how many lists are non-empty
        non_empty_count = sum(
            bool(lst) for lst in [args.locations, args.actors, args.objects])
        assert non_empty_count == 3, (
            "The number non-empty lists from locations, actors, and objects must be 3"
        )

        jsonl_dataset, txt_dataset, folder_path = proc.create_complextracking(
            q_stories=args.q_stories,
            states_qty=args.states_lenght,
            locations=args.locations,
            actors=args.actors,
            objects=args.objects,
            question=args.question,
            answer=args.answer,
            path=output_path,
            verbosity=getattr(logging, f"{args.verbosity}"),
            logger=main_logger,
            gen_kwargs=args.gen_kwargs,
        )

    else:
        raise ValueError("At least two lists must be non-empty")

    # At this point the jsonl_dataset and folder_path are defined
    # Time to save the dataset

    try:
        proc.save_as_jsonl(jsonl_dataset, folder_path, main_logger)
        proc.save_as_txt(txt_dataset, folder_path, main_logger)
    except Exception as e:
        main_logger.exception("Error", exception=e)

    end_time = time.time()
    main_logger.info("Execution time", time=end_time - start_time)


if __name__ == "__main__":
    main()
