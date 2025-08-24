import argparse

from reinforced_cot.finetune.grpo import GRPOTrainerWrapper
from reinforced_cot.finetune.sft import SupervisedFineTuning
from reinforced_cot.common.utils.params_manager import ParamsManager


def main(args):
    config = ParamsManager.parse(args.config_path)
    config = config[args.stage.upper()]

    if args.stage.lower() == "sft":
        trainer = SupervisedFineTuning(config)
    elif args.stage.lower() == "grpo":
        trainer = GRPOTrainerWrapper(config)
    else:
        raise ValueError(f"Unknown training stage: {args.stage}. Please choose from 'sft', 'ppo'.")

    trainer.train()


def merge_weights(args):
    from reinforced_cot.finetune.helper import merge_sft_adapter

    config = ParamsManager.parse(args.config_path)
    config = config[args.stage.upper()]
    merge_sft_adapter(
        base_model_path=config["model_path"],
        sft_adapter_path="/home/michael.xu2/Projects/Multimodal-Reinforce-CoT/tmp/20250822_130756_SFT/ckpt/epoch_6_loss_1.97_pass_0.48",
        output_path="/home/michael.xu2/Projects/qwen/qwen-merged-sft",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["sft", "grpo"],
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
