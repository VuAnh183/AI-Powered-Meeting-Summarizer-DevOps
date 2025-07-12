import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import whisper
import torchaudio
import librosa
import os
import re
import tqdm as notebook_tqdm
from pathlib import Path

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

torch.cuda.set_per_process_memory_fraction(0.9, device=0)


custom_model_dir = Path(r"D:/NoteSummarizer/AI-Powered-Meeting-Summarizer/models/whisper").as_posix()


# Load processor and model
ASR_model = whisper.load_model('medium', download_root=custom_model_dir)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ASR_model.to(device)


def transcribe_audio(audio_path, chunk_duration_sec=30):
    """
    Transcribe long audio using OpenAI Whisper with manual chunking.
    """
    # Load and resample audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample to 16000 Hz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    waveform = waveform.squeeze()  # mono
    total_samples = waveform.shape[0]
    chunk_size = int(sample_rate * chunk_duration_sec)

    transcriptions = []

    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        chunk = waveform[start:end].cpu().numpy()

        # Whisper expects 16-bit float PCM data
        audio_np = chunk.astype("float32")

        # Transcribe each chunk
        result = ASR_model.transcribe(audio_np, language="en")
        transcriptions.append(result["text"].strip())

        torch.cuda.empty_cache()

    return " ".join(transcriptions)
  


custom_model_dir = Path(r"D:/NoteSummarizer/AI-Powered-Meeting-Summarizer/models/meetingSum-bart").as_posix()

tokenizer = BartTokenizer.from_pretrained(custom_model_dir)
bart_model = BartForConditionalGeneration.from_pretrained(custom_model_dir)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bart_model.to(device)


def summarize_text(text, max_chunk_tokens=1024, summary_max_length=250):
    """Summarize each chunks with the BART model."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        tokens = tokenizer.encode(current_chunk + sentence, truncation=False)
        if len(tokens) <= max_chunk_tokens:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    all_summaries = []
    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=1024,  # Max input for BART-large
            truncation=True,
            padding="max_length"
        ).to(device)

        with torch.no_grad():
            summary_ids = bart_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=summary_max_length,     # 🔼 Increase this as needed (up to ~512)
                min_length=80,                     # Optional: force a minimum length
                length_penalty=2.0,                # Encourage longer summaries
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        all_summaries.append(summary.strip())

        # Free memory
        del inputs
        torch.cuda.empty_cache()

    return all_summaries
  
def summarize_chunks(chunks, summary_max_length=250):
    """Take a list of chunk summaries and generate a final summary."""
    # Join all chunk summaries into one text
    combined_text = " ".join(chunks)

    # Tokenize the combined summaries
    inputs = tokenizer(
        combined_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding="max_length"
    ).to(device)

    # Generate the final summary
    with torch.no_grad():
        summary_ids = bart_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=summary_max_length,
            min_length=80,
            length_penalty=2.0,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return final_summary.strip()