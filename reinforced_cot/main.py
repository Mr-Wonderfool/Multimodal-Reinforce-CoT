import argparse

from reinforced_cot.finetune.sft import SupervisedFineTuning
from reinforced_cot.finetune.grpo import GroupRelativePolicyOptimization
from reinforced_cot.common.utils.params_manager import ParamsManager


def main(args):
    config = ParamsManager.parse(args.config_path)
    config = config[args.stage.upper()]

    if args.stage.lower() == "sft":
        trainer = SupervisedFineTuning(config)
    elif args.stage.lower() == "grpo":
        trainer = GroupRelativePolicyOptimization(config)
    else:
        raise ValueError(f"Unknown training stage: {args.stage}. Please choose from 'sft', 'ppo'.")

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["sft", "ppo"],
        help="The training stage to run.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Absolute path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args)
