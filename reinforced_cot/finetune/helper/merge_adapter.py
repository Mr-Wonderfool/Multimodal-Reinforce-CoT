import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor


def merge_sft_adapter(base_model_path, sft_adapter_path, output_path):
    """
    Loads a base model and an SFT LoRA adapter, merges them, and saves the
    resulting full model.
    """
    print(f"Loading base model from {base_model_path}...")
    # Load the base model in a high-precision format for an accurate merge
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"Loading SFT adapter from {sft_adapter_path}...")
    sft_peft_model = PeftModel.from_pretrained(base_model, sft_adapter_path)

    print("Merging adapter weights into the base model...")
    merged_model = sft_peft_model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path)
    # Also save the processor for a self-contained checkpoint
    processor = AutoProcessor.from_pretrained(base_model_path)
    processor.save_pretrained(output_path)
    
    print("Merge complete.")