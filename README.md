Execute code 

```bash
uv run python main.py
```

```python

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import torch
from PIL import Image

base_model    = "Qwen/Qwen2.5-VL-3B-Instruct"
adapter_model = "shantipriya/odia-ocr-qwen-finetuned_v2"

processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, adapter_model)
model.eval()

def ocr_image(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": "Extract the Odia text from this image. Return only the text."}
        ]
    }]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=1.0)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

print(ocr_image("odia_word.png"))

```

"
ତ ସିଏ ମୋତେ ଜବରଦସ୍ତି ଫୋର୍ସ କଲେ କରିବା ପାଇଁକା ସେ ମୁଭିଟା ବଟ୍ ମୋ ଘରେ ଯେହେତୁ ଟିପିକାଲ୍ ମିଡିଲ୍ କ୍ଲାସ୍ ଫ୍ୟାମିଲିର ଓଡ଼ିଆ ବ୍ରାହ୍ମଣ ଘର ଫ୍ୟାମିଲି ତ ଘରେ ଓଡ଼ିଆ ଘରେ ୟୁନୋ ହିରୋଇନ ହେଲେ କଣ କିଏ ବାହା ହେବ
"