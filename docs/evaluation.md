# Evaluate Trained Models
1. We provided the trained SFT and GSPO weight under `checkpoint.zip`.
2. For any checkpoint folder with `adapter_config.json`, evaluate your model with our test set via the following command:
```bash
cd Multimodal-Reinforce-CoT/reinforced_cot/scripts
chmod 777 eval.sh
./eval.sh
```
Inside `eval.sh`, we check for the configuration file `Multimodal-Reinforce-CoT/configs/eval/eval.yaml`, notice several keys have to be present to ensure proper evaluation:
- `pipeline.test.test_file`, the test jsonl file
- `test_image_dir`, directory containing the test images
- `lora_adapter_path`, this is the path to the adapter weights, and **has to be additionally added to the configuration before evaluation**
3. The evaluation result `eval_samples_full_test.json` will be stored under `result` under the logger directory. Examples in this file looks like:
```json
{
  "image_id": "2410233",
  "instruction": "\n### Question:\nPlease answer based on the image content: Are there both cars and fences in this image?",
  "gt_answer": "No",
  "prediction_raw": "<think>\nTo determine if there are both cars and fences in the image, I need to carefully examine the elements present in the picture. The image shows an elephant standing in a grassy area with rocks and trees in the background. There is also a fence visible in the distance. However, there are no vehicles such as cars present in the image. Therefore, the answer is no, there are not both cars and fences in this image.\n</think>\n<answer>\nNo, there are not both cars and fences in this image.\n</answer>",
  "pred_think": "To determine if there are both cars and fences in the image, I need to carefully examine the elements present in the picture. The image shows an elephant standing in a grassy area with rocks and trees in the background. There is also a fence visible in the distance. However, there are no vehicles such as cars present in the image. Therefore, the answer is no, there are not both cars and fences in this image.",
  "pred_answer": "No, there are not both cars and fences in this image.",
  "is_correct": 1,
  "is_consist": 1
},
```
4. In `output.log` file, you will see the final quantitative result:
```bash
2025-08-23 17:08:28, Logger_20250823_170812_SFT, INFO: Loaded LoRA adapter from ...
2025-08-23 17:27:46, Logger_20250823_170812_SFT, INFO: Saved 500 evaluation samples to ...
2025-08-23 17:27:46, Logger_20250823_170812_SFT, INFO: Evaluation accuracy: 0.8480
2025-08-23 17:27:46, Logger_20250823_170812_SFT, INFO: Consistent samples percentage: 0.9575
2025-08-23 17:27:46, Logger_20250823_170812_SFT, INFO: [Done] Test accuracy: 0.8480
```