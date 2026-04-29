from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
from PIL import Image

model_id = "OdiaGenAIOCR/odia-ocr-qwen-finetuned-merged"

# Load processor and model
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

# Run OCR on an image
image = Image.open("data/od_note.jpg").convert("RGB")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": "Extract all Odia text from this image. Return only the text."},
    ],
}]

text_input = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(
    text=[text_input], images=[image], return_tensors="pt"
).to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

# Decode only the generated portion
input_len = inputs["input_ids"].shape[1]
result = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
print(result)
