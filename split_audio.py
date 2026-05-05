"""
Silero VAD Audio Splitter
=========================
Splits a long audio file into sequentially named audio files
(1_audio.wav, 2_audio.wav, …) using Silero VAD — one of the most
accurate open-source VAD models available.

Each output file contains a continuous speech region, capped at
--max-sec seconds (default 30). Output preserves the original format
and sample rate unless you override with --format / --sr.

Dependencies:
    pip install torch torchaudio pydub onnxruntime

FFmpeg (for mp3 / m4a / flac / ogg input-output):
    Linux : sudo apt install ffmpeg
    Mac   : brew install ffmpeg
    Win   : https://ffmpeg.org/download.html
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

# ── lazy imports (checked at runtime) ───────────────────────────────────────
try:
    import torch
    import torchaudio
except ImportError:
    sys.exit("❌  Please install torch and torchaudio:\n"
             "    pip install torch torchaudio")

try:
    from pydub import AudioSegment
except ImportError:
    sys.exit("❌  Please install pydub:\n    pip install pydub")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SILERO_REPO   = "snakers4/silero-vad"
SILERO_MODEL  = "silero_vad"
SILERO_SR     = 16_000          # model expects 16 kHz

# Tunable defaults
DEFAULT_MAX_SEC        = 30     # hard cap per output file
DEFAULT_THRESHOLD      = 0.45   # speech probability threshold (0–1)
DEFAULT_MIN_SPEECH_MS  = 250    # ignore speech bursts shorter than this
DEFAULT_MIN_SILENCE_MS = 400    # silence gap needed to split segments
DEFAULT_PADDING_MS     = 200    # extra ms added before/after each segment


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str):
    """
    Load any audio file with torchaudio (+ ffmpeg fallback via pydub).
    Returns (waveform_tensor [1, samples], original_sample_rate, pydub_audio).
    """
    ext = os.path.splitext(path)[1].lower()

    # torchaudio handles wav/flac natively; everything else needs ffmpeg
    try:
        waveform, sr = torchaudio.load(path)
    except Exception:
        # Fallback: decode via pydub → write temp wav → reload
        tmp_wav = "__tmp_input__.wav"
        seg = AudioSegment.from_file(path)
        seg.export(tmp_wav, format="wav")
        waveform, sr = torchaudio.load(tmp_wav)
        os.remove(tmp_wav)

    # Keep original pydub object for high-quality export later
    try:
        pydub_audio = AudioSegment.from_file(path)
    except Exception:
        # reconstruct from tensor if pydub can't read directly
        pydub_audio = _tensor_to_pydub(waveform, sr)

    return waveform, sr, pydub_audio


def _tensor_to_pydub(waveform: torch.Tensor, sr: int) -> AudioSegment:
    """Convert a torch waveform tensor to a pydub AudioSegment."""
    import io, wave, struct
    mono = waveform.mean(dim=0).numpy()
    pcm  = (mono * 32767).clip(-32768, 32767).astype("int16")
    buf  = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return AudioSegment.from_wav(buf)


def resample_for_vad(waveform: torch.Tensor, src_sr: int) -> torch.Tensor:
    """Resample to 16 kHz mono for Silero."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if src_sr != SILERO_SR:
        waveform = torchaudio.functional.resample(waveform, src_sr, SILERO_SR)
    return waveform.squeeze(0)   # [samples]


# ─────────────────────────────────────────────────────────────────────────────
# SILERO VAD
# ─────────────────────────────────────────────────────────────────────────────

def load_silero():
    """Download (cached) and return the Silero VAD model + get_speech_ts utility."""
    print("  Loading Silero VAD model (downloaded once, cached locally) …")
    model, utils = torch.hub.load(
        repo_or_dir = SILERO_REPO,
        model       = SILERO_MODEL,
        force_reload= False,
        onnx        = False,
        verbose     = False,
    )
    get_speech_ts = utils[0]   # get_speech_timestamps
    return model, get_speech_ts


def run_vad(model, get_speech_ts, audio_1d: torch.Tensor,
            threshold: float      = DEFAULT_THRESHOLD,
            min_speech_ms: int    = DEFAULT_MIN_SPEECH_MS,
            min_silence_ms: int   = DEFAULT_MIN_SILENCE_MS,
            padding_ms: int       = DEFAULT_PADDING_MS) -> list[dict]:
    """
    Run Silero VAD and return a list of {'start': samples, 'end': samples}.
    All timing is in *samples* at SILERO_SR (16 kHz).
    """
    timestamps = get_speech_ts(
        audio_1d,
        model,
        sampling_rate         = SILERO_SR,
        threshold             = threshold,
        min_speech_duration_ms= min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms         = padding_ms,
        return_seconds        = False,   # keep in samples
    )
    return timestamps   # list of {'start': int, 'end': int}


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENT MERGING & SPLITTING
# ─────────────────────────────────────────────────────────────────────────────

def samples_to_ms(samples: int, sr: int = SILERO_SR) -> float:
    return samples / sr * 1000


def merge_and_cap(timestamps: list[dict], max_ms: int,
                  src_sr: int, orig_duration_ms: int) -> list[tuple[int, int]]:
    """
    Convert VAD sample-level timestamps → millisecond (start, end) tuples
    in the *original* audio's time-base (src_sr), then:
      1. Merge consecutive segments that fit within max_ms together.
      2. Hard-split any single segment that still exceeds max_ms.

    Returns list of (start_ms, end_ms) in original audio time.
    """
    if not timestamps:
        return []

    # Convert VAD samples (16 kHz) → ms, then scale to original sr ms
    scale = src_sr / SILERO_SR   # unused for ms math, but kept for clarity

    def vad_samples_to_orig_ms(s):
        # VAD samples → seconds → ms (same regardless of src_sr because VAD
        # always operates at SILERO_SR, and ms is ms)
        return s / SILERO_SR * 1000

    segs = [(vad_samples_to_orig_ms(t["start"]),
             vad_samples_to_orig_ms(t["end"]))
            for t in timestamps]

    # 1. Greedy merge
    merged = []
    cur_s, cur_e = segs[0]
    for s, e in segs[1:]:
        if e - cur_s <= max_ms:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # 2. Hard-split oversized
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


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_chunks(pydub_audio: AudioSegment,
                  chunks: list[tuple[int, int]],
                  output_dir: str,
                  base_name: str,
                  fmt: str):
    """
    Slice pydub_audio and save each chunk as:
        {output_dir}/{idx}_{base_name}.{fmt}
    e.g.  output/1_audio.wav, output/2_audio.wav, …
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    for idx, (start_ms, end_ms) in enumerate(chunks, start=1):
        segment  = pydub_audio[start_ms:end_ms]
        filename = f"{idx}_{base_name}.{fmt}"
        out_path = os.path.join(output_dir, filename)

        export_kwargs = {"format": fmt}
        if fmt == "mp3":
            export_kwargs["bitrate"] = "192k"

        segment.export(out_path, **export_kwargs)
        dur = (end_ms - start_ms) / 1000
        saved.append(out_path)
        print(f"    [{idx:>4}]  {start_ms/1000:8.2f}s → {end_ms/1000:8.2f}s"
              f"  ({dur:5.2f}s)  →  {filename}")

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def split_audio(
    input_path      : str,
    output_dir      : str  = "output",
    base_name       : str  = "audio",          # used in filenames: 1_audio.wav
    max_chunk_sec   : int  = DEFAULT_MAX_SEC,
    threshold       : float= DEFAULT_THRESHOLD,
    min_speech_ms   : int  = DEFAULT_MIN_SPEECH_MS,
    min_silence_ms  : int  = DEFAULT_MIN_SILENCE_MS,
    padding_ms      : int  = DEFAULT_PADDING_MS,
    output_format   : str  = None,             # None = keep original format
) -> list[str]:
    """
    Split *input_path* into sequentially named speech segments.

    Returns
    -------
    List of output file paths.
    """
    # ── resolve output format ──────────────────────────────────────────────
    src_ext = os.path.splitext(input_path)[1].lstrip(".").lower() or "wav"
    fmt     = output_format or src_ext
    if fmt == "m4a":
        fmt = "ipod"   # pydub codec name for m4a/aac

    max_ms  = max_chunk_sec * 1000

    # ── load ───────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading:  {input_path}")
    waveform, src_sr, pydub_audio = load_audio(input_path)
    total_sec = len(pydub_audio) / 1000
    print(f"      Duration : {total_sec:.1f}s  |  "
          f"Sample rate: {src_sr} Hz  |  "
          f"Channels: {waveform.shape[0]}")

    # ── silero ────────────────────────────────────────────────────────────
    print(f"\n[2/4] Running Silero VAD …")
    print(f"      threshold={threshold}  min_speech={min_speech_ms}ms  "
          f"min_silence={min_silence_ms}ms  padding={padding_ms}ms")
    model, get_speech_ts = load_silero()
    audio_16k = resample_for_vad(waveform, src_sr)
    timestamps = run_vad(model, get_speech_ts, audio_16k,
                         threshold=threshold,
                         min_speech_ms=min_speech_ms,
                         min_silence_ms=min_silence_ms,
                         padding_ms=padding_ms)
    print(f"      Raw speech segments detected: {len(timestamps)}")

    if not timestamps:
        print("\n⚠️  No speech detected. Try lowering --threshold.")
        return []

    # ── merge & cap ────────────────────────────────────────────────────────
    print(f"\n[3/4] Merging segments (max {max_chunk_sec}s per file) …")
    chunks = merge_and_cap(timestamps, max_ms, src_sr, len(pydub_audio))
    print(f"      Output files to create: {len(chunks)}")

    # ── export ─────────────────────────────────────────────────────────────
    print(f"\n[4/4] Exporting to '{output_dir}/' as .{fmt} …\n")
    saved = export_chunks(pydub_audio, chunks, output_dir, base_name, fmt)

    total_speech = sum(e - s for s, e in chunks) / 1000
    print(f"\n✅  Done!  {len(saved)} file(s) saved in '{output_dir}/'")
    print(f"    Total speech captured : {total_speech:.1f}s  "
          f"/ {total_sec:.1f}s source")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Silero VAD — split long audio into sequentially named speech files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",
                   help="Input audio file (wav, mp3, flac, m4a, ogg …)")
    p.add_argument("-o", "--output-dir", default="output",
                   help="Folder to write output files into")
    p.add_argument("-n", "--name", default="audio",
                   help="Base name used in output files: 1_NAME.ext, 2_NAME.ext …")
    p.add_argument("-m", "--max-sec", type=int, default=DEFAULT_MAX_SEC,
                   help="Maximum duration per output file (seconds)")
    p.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Speech probability threshold 0–1 (lower = more sensitive)")
    p.add_argument("--min-speech", type=int, default=DEFAULT_MIN_SPEECH_MS,
                   help="Minimum speech burst duration to keep (ms)")
    p.add_argument("--min-silence", type=int, default=DEFAULT_MIN_SILENCE_MS,
                   help="Minimum silence gap between segments (ms)")
    p.add_argument("--padding", type=int, default=DEFAULT_PADDING_MS,
                   help="Extra audio padding around each segment (ms)")
    p.add_argument("-f", "--format", default=None,
                   help="Output format override: wav / mp3 / flac / ogg  "
                        "(default: same as input)")
    args = p.parse_args()

    split_audio(
        input_path    = args.input,
        output_dir    = args.output_dir,
        base_name     = args.name,
        max_chunk_sec = args.max_sec,
        threshold     = args.threshold,
        min_speech_ms = args.min_speech,
        min_silence_ms= args.min_silence,
        padding_ms    = args.padding,
        output_format = args.format,
    )


split_audio("audio_full.flac")