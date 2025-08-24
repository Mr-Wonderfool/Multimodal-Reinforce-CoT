import argparse

from reinforced_cot.finetune.helper import Evaluator
from reinforced_cot.common.utils.params_manager import ParamsManager


def main(args):
    config = ParamsManager.parse(args.config_path)

    evaluator = Evaluator(config=config)
    evaluator.evaluate(tag=args.stage.upper())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["sft", "grpo"],
        help="The evaluated model name.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Absolute path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args)
