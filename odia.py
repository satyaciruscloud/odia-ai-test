from dotenv import load_dotenv
from transformers import AutoModel
import torch, torchaudio
from silero_vad import load_silero_vad
from silero_vad import get_speech_timestamps
from tqdm import tqdm
from docx import Document

load_dotenv()

import os


os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Load the model
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)



# Load Audio + Resample to 16 kHz
def load_audio(file_path, target_sr=16000):
    try:
        wav, sr = torchaudio.load(file_path)
    except ImportError as exc:
        if "TorchCodec is required" not in str(exc):
            raise

        # Fallback for environments where torchaudio requires torchcodec.
        import soundfile as sf

        wav_np, sr = sf.read(file_path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav_np.T)

    # Convert to mono
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=target_sr
        )
        wav = resampler(wav)
        sr = target_sr

    return wav, sr

vad_model = load_silero_vad()


# Get Speech Segments Using VAD
def get_vad_segments(wav, sr):

    timestamps = get_speech_timestamps(
        wav.squeeze(),
        vad_model,
        sampling_rate=sr,
    )

    print("Number of speech segments:", len(timestamps))

    return timestamps

# Split Long Segments into ≤15 sec Chunks
def split_segments_into_chunks(
    wav,
    segments,
    sr,
    max_duration=15,
    overlap=1
):

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

    print("Total chunks created:", len(chunks))

    return chunks

# Transcription Function
def transcribe_chunks(
    model,
    chunks,
    language="or",
    decoding="rnnt",
    batch_size=4
):

    transcripts = []

    for i in tqdm(range(0, len(chunks), batch_size)):

        batch = chunks[i:i + batch_size]

        with torch.no_grad():

            results = []

            for chunk in batch:

                text = model(
                    chunk,
                    language,
                    decoding
                )

                results.append(text)

        transcripts.extend(results)

    return transcripts

# Merge Final Transcript
def merge_transcripts(transcripts):

    final_text = " ".join(transcripts)

    return final_text


def save_transcript_to_txt(transcript_text, audio_file_path):
    # Get base name of the audio file (without extension)
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    
    # Create output file path with .txt extension
    output_path = f"{base_name}_transcript.txt"

    # Write transcript to text file
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("Odia Transcript\n\n")
        file.write(transcript_text)

    return output_path



# Full Pipeline Execution
# audio_path = "type.flac"
audio_path = "audio_full.flac"
# audio_path = "audio32type.flac"

# Load audio
wav, sr = load_audio(audio_path)

# Get speech regions
segments = get_vad_segments(wav, sr)

# Create chunks
chunks = split_segments_into_chunks(
    wav,
    segments,
    sr,
    max_duration=15,
    overlap=1
)

# Run transcription
transcripts = transcribe_chunks(
    model=model,
    chunks=chunks,
    language="or",
    decoding="rnnt",
    batch_size=4
)

# Final output
final_transcript = merge_transcripts(transcripts)

print("\nFINAL TRANSCRIPT:\n")
print(final_transcript)

docx_path = save_transcript_to_txt(final_transcript, audio_path)
print(f"\nTranscript saved to Word file: {docx_path}")