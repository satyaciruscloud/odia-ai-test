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


```shell
uv run python split_audio_with_vad.py audio_full.flac -o output_splits/ -n audio --max-sec 30
```

### Split & Transcribe (Parallel)

Splits audio into speech segments, transcribes each in parallel, and saves aggregated output in sequence order.

```shell
uv run python split_and_transcribe.py audio_full.flac
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `-o`, `--output-dir` | Directory for split audio files | `output_splits` |
| `-m`, `--max-sec` | Max duration per split file (seconds) | `30` |
| `-t`, `--threshold` | VAD speech detection threshold (0–1) | `0.45` |
| `-w`, `--workers` | Parallel transcription workers | `4` |
| `-f`, `--format` | Output format override (`wav`, `mp3`, `flac`) | same as input |

**Example with all options:**

```shell
uv run python split_and_transcribe.py audio_full.flac -o output_splits -m 30 -t 0.45 -w 4 -f wav
```

Output: `<input_name>_full_transcript.txt` with each segment labeled `[1]`, `[2]`, `[3]`, etc.

"


**Sequential (default):**
```bash
uv run python split_and_transcribe.py audio_full.flac -o output_splits/ -n audio --max-sec 30
```

**Parallel:**
```bash
uv run python split_and_transcribe.py audio_full.flac -p --workers 4
```

**Key design decisions:**

**Sequential order guarantee** — in both modes, results go into a pre-allocated `transcripts[idx]` array. Parallel mode uses `as_completed()` (unordered by arrival) but writes to the correct index, so the final merge is always chunk 1 → 2 → 3 → N.

**Parallel mode caveat** — the indic-conformer model isn't thread-safe, so parallel mode serializes inference behind a `Lock`. This means you won't get a GPU speedup from parallelism, but it's useful if I/O (chunk loading) is your bottleneck. If you have multiple GPUs, a `ProcessPoolExecutor` approach with one model per process would actually parallelize inference — let me know if you want that.

**Output file** — saved at `output_splits/{base}_transcript.txt` with both per-chunk labels and a full merged transcript at the bottom.

**`--lang` flag** — you can swap `or` → `hi`, `bn`, `te` etc. for other Indic languages without touching code.