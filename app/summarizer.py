from app.utils import transcribe_audio, summarize_text, summarize_chunks

def summarize_meeting(file_path: str) -> str:
    transcript = transcribe_audio(file_path)
    summary = summarize_text(transcript)
    final_summary = summarize_chunks(summary)
    return final_summary
