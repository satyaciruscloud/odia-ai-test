"""
Split + Transcribe Pipeline (Odia / any Indic language)
========================================================
1. Splits audio via Silero VAD  →  output_dir/1_audio.wav, 2_audio.wav …
2. Transcribes each chunk with  ai4bharat/indic-conformer-600m-multilingual
3. Merges transcripts in sequential order regardless of processing mode.

Usage examples
--------------
# Sequential (default)
uv run python split_and_transcribe.py audio_full.flac

# Parallel  (uses all CPU-visible CUDA devices / threads)
uv run python split_and_transcribe.py audio_full.flac -p

# Full control
uv run python split_and_transcribe.py audio_full.flac \
    -o output_splits/ -n audio --max-sec 30 \
    --lang or --decoding rnnt --batch-size 4 \
    -p --workers 4

Dependencies
------------
    pip install torch torchaudio transformers silero-vad pydub python-docx tqdm
    (+ ffmpeg on PATH for mp3/flac/m4a input)
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ── third-party ─────────────────────────────────────────────────────────────
try:
    import torch
    import torchaudio
except ImportError:
    sys.exit("❌  pip install torch torchaudio")

try:
    from pydub import AudioSegment
except ImportError:
    sys.exit("❌  pip install pydub")

try:
    from transformers import AutoModel
except ImportError:
    sys.exit("❌  pip install transformers")

try:
    from silero_vad import load_silero_vad, get_speech_timestamps
except ImportError:
    sys.exit("❌  pip install silero-vad")

from tqdm import tqdm


# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS / DEFAULTS
# ════════════════════════════════════════════════════════════════════════════
SILERO_SR              = 16_000
DEFAULT_MAX_SEC        = 30
DEFAULT_THRESHOLD      = 0.45
DEFAULT_MIN_SPEECH_MS  = 250
DEFAULT_MIN_SILENCE_MS = 400
DEFAULT_PADDING_MS     = 200
DEFAULT_LANG           = "or"
DEFAULT_DECODING       = "rnnt"
DEFAULT_BATCH_SIZE     = 4


# ════════════════════════════════════════════════════════════════════════════
#  AUDIO LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_audio_torchaudio(path: str):
    """Load audio → (waveform [C, T], sr, pydub_audio)."""
    try:
        waveform, sr = torchaudio.load(path)
    except Exception:
        tmp = "__tmp_input__.wav"
        AudioSegment.from_file(path).export(tmp, format="wav")
        waveform, sr = torchaudio.load(tmp)
        os.remove(tmp)

    try:
        pydub_audio = AudioSegment.from_file(path)
    except Exception:
        pydub_audio = _tensor_to_pydub(waveform, sr)

    return waveform, sr, pydub_audio


def _tensor_to_pydub(waveform: torch.Tensor, sr: int) -> AudioSegment:
    import io, wave
    mono = waveform.mean(dim=0).numpy()
    pcm  = (mono * 32767).clip(-32768, 32767).astype("int16")
    buf  = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(sr); wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return AudioSegment.from_wav(buf)


def resample_for_vad(waveform: torch.Tensor, src_sr: int) -> torch.Tensor:
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if src_sr != SILERO_SR:
        waveform = torchaudio.functional.resample(waveform, src_sr, SILERO_SR)
    return waveform.squeeze(0)


def load_chunk_as_tensor(path: str, target_sr: int = SILERO_SR) -> torch.Tensor:
    """Load a saved chunk file back as a tensor for transcription."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


# ════════════════════════════════════════════════════════════════════════════
#  VAD
# ════════════════════════════════════════════════════════════════════════════

def run_vad(audio_1d: torch.Tensor,
            vad_model,
            threshold: float,
            min_speech_ms: int,
            min_silence_ms: int,
            padding_ms: int) -> list[dict]:
    return get_speech_timestamps(
        audio_1d,
        vad_model,
        sampling_rate            = SILERO_SR,
        threshold                = threshold,
        min_speech_duration_ms   = min_speech_ms,
        min_silence_duration_ms  = min_silence_ms,
        speech_pad_ms            = padding_ms,
        return_seconds           = False,
    )


def merge_and_cap(timestamps: list[dict],
                  max_ms: int,
                  src_sr: int) -> list[tuple[int, int]]:
    if not timestamps:
        return []

    def to_ms(s):
        return s / SILERO_SR * 1000

    segs = [(to_ms(t["start"]), to_ms(t["end"])) for t in timestamps]

    # greedy merge
    merged, (cur_s, cur_e) = [], segs[0]
    for s, e in segs[1:]:
        if e - cur_s <= max_ms:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # hard split oversized
    final = []
    for s, e in merged:
        dur = e - s
        if dur <= max_ms:
            final.append((int(s), int(e)))
        else:
            pos = s
            while pos < e:
                final.append((int(pos), int(min(pos + max_ms, e))))
                pos += max_ms
    return final


# ════════════════════════════════════════════════════════════════════════════
#  SPLIT → SAVE CHUNKS
# ════════════════════════════════════════════════════════════════════════════

def export_chunks(pydub_audio: AudioSegment,
                  chunks: list[tuple[int, int]],
                  output_dir: str,
                  base_name: str,
                  fmt: str) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for idx, (s_ms, e_ms) in enumerate(chunks, start=1):
        seg      = pydub_audio[s_ms:e_ms]
        fname    = f"{idx}_{base_name}.{fmt}"
        out_path = os.path.join(output_dir, fname)
        kw       = {"format": fmt}
        if fmt == "mp3":
            kw["bitrate"] = "192k"
        seg.export(out_path, **kw)
        dur = (e_ms - s_ms) / 1000
        print(f"    [{idx:>4}]  {s_ms/1000:8.2f}s → {e_ms/1000:8.2f}s"
              f"  ({dur:5.2f}s)  →  {fname}")
        saved.append(out_path)
    return saved


# ════════════════════════════════════════════════════════════════════════════
#  TRANSCRIPTION
# ════════════════════════════════════════════════════════════════════════════

def transcribe_single(model, chunk_path: str, lang: str, decoding: str) -> str:
    """Transcribe one saved chunk file → string."""
    wav = load_chunk_as_tensor(chunk_path)
    with torch.no_grad():
        text = model(wav, lang, decoding)
    return text if isinstance(text, str) else str(text)


def transcribe_sequential(model,
                           chunk_paths: list[str],
                           lang: str,
                           decoding: str,
                           batch_size: int) -> list[str]:
    """
    Process chunks in order, batch_size at a time.
    Returns list of transcripts aligned with chunk_paths.
    """
    transcripts = [""] * len(chunk_paths)

    for i in tqdm(range(0, len(chunk_paths), batch_size),
                  desc="Transcribing (sequential)", unit="batch"):
        batch = chunk_paths[i : i + batch_size]
        for j, path in enumerate(batch):
            transcripts[i + j] = transcribe_single(model, path, lang, decoding)

    return transcripts


def transcribe_parallel(model,
                        chunk_paths: list[str],
                        lang: str,
                        decoding: str,
                        max_workers: int) -> list[str]:
    """
    Submit all chunks to a thread pool; collect results indexed by
    original position → sequential order guaranteed in output.
    """
    # Pre-allocate results array so order is always correct
    transcripts = [""] * len(chunk_paths)
    model_lock  = Lock()   # indic-conformer is not thread-safe internally

    def _worker(idx: int, path: str) -> tuple[int, str]:
        with model_lock:            # serialize actual inference
            text = transcribe_single(model, path, lang, decoding)
        return idx, text

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_worker, idx, path): idx
            for idx, path in enumerate(chunk_paths)
        }
        pbar = tqdm(total=len(chunk_paths),
                    desc="Transcribing (parallel)", unit="chunk")
        for fut in as_completed(futures):
            idx, text = fut.result()
            transcripts[idx] = text
            pbar.update(1)
        pbar.close()

    return transcripts


# ════════════════════════════════════════════════════════════════════════════
#  OUTPUT WRITERS
# ════════════════════════════════════════════════════════════════════════════

def save_transcript(transcripts: list[str],
                    chunk_paths: list[str],
                    audio_path: str,
                    output_dir: str,
                    lang: str) -> str:
    """
    Save a .txt transcript next to the output_dir.
    Each chunk gets its own labelled block, then a full merged section.
    """
    base   = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = os.path.join(output_dir, f"{base}_transcript.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Transcript  |  lang={lang}  |  source={audio_path}\n")
        f.write("=" * 70 + "\n\n")

        f.write("── Per-chunk transcripts ──\n\n")
        for i, (path, text) in enumerate(zip(chunk_paths, transcripts), start=1):
            chunk_name = os.path.basename(path)
            f.write(f"[{i:>4}] {chunk_name}\n")
            f.write(text.strip() + "\n\n")

        f.write("\n" + "─" * 70 + "\n")
        f.write("── Full merged transcript ──\n\n")
        f.write(" ".join(t.strip() for t in transcripts) + "\n")

    return out_path


# ════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    # ── resolve output format ────────────────────────────────────────────
    src_ext = os.path.splitext(args.input)[1].lstrip(".").lower() or "wav"
    fmt     = args.format or src_ext
    if fmt == "m4a":
        fmt = "ipod"
    max_ms = args.max_sec * 1000

    # ── 1. Load audio ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[1/4] Loading audio: {args.input}")
    waveform, src_sr, pydub_audio = load_audio_torchaudio(args.input)
    total_sec = len(pydub_audio) / 1000
    print(f"      Duration: {total_sec:.1f}s  |  SR: {src_sr} Hz  |  "
          f"Channels: {waveform.shape[0]}")

    # ── 2. VAD ───────────────────────────────────────────────────────────
    print(f"\n[2/4] Running Silero VAD …")
    vad_model = load_silero_vad()
    audio_16k  = resample_for_vad(waveform, src_sr)
    timestamps = run_vad(audio_16k, vad_model,
                         threshold      = args.threshold,
                         min_speech_ms  = args.min_speech,
                         min_silence_ms = args.min_silence,
                         padding_ms     = args.padding)
    print(f"      Raw speech segments: {len(timestamps)}")

    if not timestamps:
        print("⚠️  No speech detected. Try lowering --threshold.")
        return

    chunks = merge_and_cap(timestamps, max_ms, src_sr)
    print(f"      After merge+cap ({args.max_sec}s): {len(chunks)} chunks")

    # ── 3. Export chunks ─────────────────────────────────────────────────
    print(f"\n[3/4] Exporting chunks to '{args.output_dir}/' …\n")
    chunk_paths = export_chunks(pydub_audio, chunks,
                                args.output_dir, args.name, fmt)

    # ── 4. Load ASR model ─────────────────────────────────────────────────
    print(f"\n[4/5] Loading ASR model (ai4bharat/indic-conformer-600m-multilingual) …")
    from dotenv import load_dotenv
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    asr_model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True
    )
    print(f"      Model loaded.  lang={args.lang}  decoding={args.decoding}")

    # ── 5. Transcribe ────────────────────────────────────────────────────
    mode = "parallel" if args.parallel else "sequential"
    print(f"\n[5/5] Transcribing {len(chunk_paths)} chunks ({mode}) …\n")

    if args.parallel:
        transcripts = transcribe_parallel(
            asr_model, chunk_paths,
            lang=args.lang, decoding=args.decoding,
            max_workers=args.workers
        )
    else:
        transcripts = transcribe_sequential(
            asr_model, chunk_paths,
            lang=args.lang, decoding=args.decoding,
            batch_size=args.batch_size
        )

    # ── Save transcript ───────────────────────────────────────────────────
    out_txt = save_transcript(transcripts, chunk_paths,
                              args.input, args.output_dir, args.lang)

    full_text = " ".join(t.strip() for t in transcripts)
    print(f"\n{'='*60}")
    print("FINAL TRANSCRIPT (preview, first 500 chars):\n")
    print(full_text[:500] + ("…" if len(full_text) > 500 else ""))
    print(f"\n✅  Transcript saved → {out_txt}")
    print(f"    Chunks saved      → {args.output_dir}/")
    print(f"    Total chunks      : {len(chunk_paths)}")


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="VAD-split + Indic ASR transcription pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── input / output ────────────────────────────────────────────────────
    p.add_argument("input",
                   help="Input audio file (wav, flac, mp3, m4a …)")
    p.add_argument("-o", "--output-dir", default="output_splits",
                   help="Folder for chunk files + transcript")
    p.add_argument("-n", "--name", default="audio",
                   help="Base name for chunk files: 1_NAME.ext, 2_NAME.ext …")
    p.add_argument("-f", "--format", default=None,
                   help="Output audio format override (wav/mp3/flac/ogg). "
                        "Default: same as input.")

    # ── VAD ───────────────────────────────────────────────────────────────
    vad = p.add_argument_group("VAD options")
    vad.add_argument("--max-sec", type=int, default=DEFAULT_MAX_SEC,
                     help="Max chunk duration (seconds)")
    vad.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                     help="Speech probability threshold 0–1")
    vad.add_argument("--min-speech", type=int, default=DEFAULT_MIN_SPEECH_MS,
                     help="Min speech burst to keep (ms)")
    vad.add_argument("--min-silence", type=int, default=DEFAULT_MIN_SILENCE_MS,
                     help="Min silence gap to split on (ms)")
    vad.add_argument("--padding", type=int, default=DEFAULT_PADDING_MS,
                     help="Padding added around each segment (ms)")

    # ── ASR ───────────────────────────────────────────────────────────────
    asr = p.add_argument_group("ASR options")
    asr.add_argument("--lang", default=DEFAULT_LANG,
                     help="Language code (or=Odia, hi=Hindi, bn=Bengali …)")
    asr.add_argument("--decoding", default=DEFAULT_DECODING,
                     choices=["rnnt", "ctc"],
                     help="Decoding strategy")
    asr.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                     help="Batch size for sequential mode")

    # ── parallelism ───────────────────────────────────────────────────────
    par = p.add_argument_group("Parallelism")
    par.add_argument("-p", "--parallel", action="store_true",
                     help="Enable parallel transcription (flag; omit for sequential)")
    par.add_argument("--workers", type=int, default=4,
                     help="Number of worker threads in parallel mode")

    return p


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    run_pipeline(args)