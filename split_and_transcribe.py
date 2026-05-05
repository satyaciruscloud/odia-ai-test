"""
Audio Split & Transcribe Pipeline
==================================
Splits long audio using Silero VAD, transcribes chunks in parallel
using AI4Bharat Indic Conformer, and aggregates results sequentially.

Dependencies:
    pip install torch torchaudio pydub onnxruntime transformers silero-vad tqdm python-dotenv
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from transformers import AutoModel
import torch
import torchaudio
from silero_vad import load_silero_vad, get_speech_timestamps
from tqdm import tqdm

from split_audio_with_vad import split_audio

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# ── Constants ───────────────────────────────────────────────────────────────
TRANSCRIPTION_SR = 16000
MAX_DURATION = 15
OVERLAP = 1
LANGUAGE = "or"
DECODING = "rnnt"
BATCH_SIZE = 4
MAX_WORKERS = 4

# ── Model loading (once, shared across threads) ─────────────────────────────
print("Loading AI4Bharat Indic Conformer model...")
transcribe_model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    trust_remote_code=True,
)
print("Loading Silero VAD model...")
vad_model = load_silero_vad()
print("Models loaded.\n")


# ── Audio loading ───────────────────────────────────────────────────────────
def load_audio(file_path: str, target_sr: int = TRANSCRIPTION_SR):
    try:
        wav, sr = torchaudio.load(file_path)
    except ImportError as exc:
        if "TorchCodec is required" not in str(exc):
            raise
        import soundfile as sf
        wav_np, sr = sf.read(file_path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav_np.T)

    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
        sr = target_sr

    return wav, sr


# ── VAD segmentation ────────────────────────────────────────────────────────
def get_vad_segments(wav, sr):
    timestamps = get_speech_timestamps(
        wav.squeeze(),
        vad_model,
        sampling_rate=sr,
    )
    return timestamps


# ── Chunk splitting ─────────────────────────────────────────────────────────
def split_into_chunks(wav, segments, sr, max_duration=MAX_DURATION, overlap=OVERLAP):
    max_samples = sr * max_duration
    overlap_samples = sr * overlap

    chunks = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        current = start
        while current < end:
            chunk_end = min(current + max_samples, end)
            chunk = wav[:, current:chunk_end]
            chunks.append(chunk)
            current += max_samples - overlap_samples

    return chunks


# ── Transcription (thread-safe) ─────────────────────────────────────────────
def transcribe_batch(batch_chunks):
    results = []
    with torch.no_grad():
        for chunk in batch_chunks:
            text = transcribe_model(chunk, LANGUAGE, DECODING)
            results.append(text)
    return results


def transcribe_all(chunks, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS):
    ordered_results = [None] * len(chunks)

    indexed_batches = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        indexed_batches.append((i // batch_size, i, batch))

    def process_batch(idx, start, batch):
        texts = transcribe_batch(batch)
        return idx, start, texts

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_batch, idx, start, batch): (idx, start)
            for idx, start, batch in indexed_batches
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Transcribing"):
            idx, start = futures[future]
            _, _, texts = future.result()
            for j, text in enumerate(texts):
                ordered_results[start + j] = text

    return ordered_results


# ── Full pipeline ───────────────────────────────────────────────────────────
def split_and_transcribe(
    input_path: str,
    output_dir: str = "output_splits",
    max_chunk_sec: int = 30,
    threshold: float = 0.45,
    min_speech_ms: int = 250,
    min_silence_ms: int = 400,
    padding_ms: int = 200,
    output_format: str = None,
    max_workers: int = MAX_WORKERS,
) -> str:
    """
    Split audio into speech segments, transcribe in parallel, and save
    aggregated transcript to a text file.

    Returns path to the transcript file.
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    transcript_path = f"{base_name}_full_transcript.txt"

    # Step 1: Split audio
    print("=" * 60)
    print("STEP 1: Splitting audio into speech segments...")
    print("=" * 60)
    split_files = split_audio(
        input_path=input_path,
        output_dir=output_dir,
        base_name="audio",
        max_chunk_sec=max_chunk_sec,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        padding_ms=padding_ms,
        output_format=output_format,
    )

    if not split_files:
        print("No speech segments found. Exiting.")
        return ""

    print(f"\nTotal split files: {len(split_files)}")

    # Step 2: Transcribe each split file in parallel
    print("\n" + "=" * 60)
    print("STEP 2: Transcribing split files in parallel...")
    print("=" * 60)

    split_files_with_idx = list(enumerate(split_files, start=1))

    def transcribe_single(idx, file_path):
        wav, sr = load_audio(file_path)
        segments = get_vad_segments(wav, sr)
        if not segments:
            return idx, ""
        chunks = split_into_chunks(wav, segments, sr)
        if not chunks:
            return idx, ""
        texts = transcribe_all(chunks, batch_size=BATCH_SIZE, max_workers=max_workers)
        return idx, " ".join(texts)

    ordered_transcripts = [None] * len(split_files)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(transcribe_single, idx, fp): idx
            for idx, fp in split_files_with_idx
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            idx, text = future.result()
            ordered_transcripts[idx - 1] = text

    # Step 3: Aggregate sequentially & save
    print("\n" + "=" * 60)
    print("STEP 3: Aggregating transcripts in sequence order...")
    print("=" * 60)

    full_transcript_lines = []
    for idx, text in enumerate(ordered_transcripts, start=1):
        full_transcript_lines.append(f"[{idx}] {text}")

    full_transcript = "\n\n".join(full_transcript_lines)

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write("Odia Transcript\n")
        f.write("=" * 60 + "\n")
        f.write(f"Source: {input_path}\n")
        f.write(f"Segments: {len(split_files)}\n")
        f.write("=" * 60 + "\n\n")
        f.write(full_transcript)
        f.write("\n")

    print(f"\nTranscript saved to: {transcript_path}")
    return transcript_path


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split audio with VAD and transcribe in parallel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("-o", "--output-dir", default="output_splits", help="Split output directory")
    parser.add_argument("-m", "--max-sec", type=int, default=30, help="Max duration per split file (seconds)")
    parser.add_argument("-t", "--threshold", type=float, default=0.45, help="VAD threshold")
    parser.add_argument("-w", "--workers", type=int, default=MAX_WORKERS, help="Parallel workers for transcription")
    parser.add_argument("-f", "--format", default=None, help="Output format override (wav/mp3/flac)")

    args = parser.parse_args()

    start_time = time.time()
    result = split_and_transcribe(
        input_path=args.input,
        output_dir=args.output_dir,
        max_chunk_sec=args.max_sec,
        threshold=args.threshold,
        max_workers=args.workers,
        output_format=args.format,
    )
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
