import gradio as gr
import torch
from utils import transcribe_audio, summarize_text, summarize_chunks

# Optional: Clean wrapper around both transcription + summarization
def summarize_meeting(file_path):
    print(f"📥 Received file: {file_path}")

    print("🔊 Transcribing audio...")
    transcript = transcribe_audio(file_path)

    print("📝 Generating summary...")
    summary = summarize_text(transcript)
    final_summary = summarize_chunks(summary)
    
    print("✅ Done!")
    return final_summary

# Define Gradio interface
demo = gr.Interface(
    fn=summarize_meeting,
    inputs=gr.File(label="Upload your meeting audio (.wav)"),
    outputs=gr.Textbox(label="Meeting Summary"),
    title="AI-Powered Meeting Summarizer",
    description="Upload a meeting audio file. The app transcribes it using Whisper and generates a summary with a fine-tuned version of BART for meeting summary.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(share=False)
