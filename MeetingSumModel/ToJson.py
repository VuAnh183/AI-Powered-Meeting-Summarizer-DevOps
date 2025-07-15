import os
import json

# Your dataset root directory
dataset_root = "./amicorpus/"
output_file = "./data/amicorpus.jsonl"

with open(output_file, "w", encoding="utf-8") as outfile:
    for folder in os.listdir(dataset_root):
        folder_path = os.path.join(dataset_root, folder)
        if os.path.isdir(folder_path):
            base_name = folder  # e.g., ES2002a
            transcript_path = os.path.join(folder_path, f"{base_name}.Mix-Headset.txt")
            summary_path = os.path.join(folder_path, f"{base_name}.summary.txt")

            # Ensure both transcript and summary exist
            if os.path.exists(transcript_path) and os.path.exists(summary_path):
                with open(transcript_path, "r", encoding="utf-8") as t_file:
                    transcript = t_file.read().strip()
                with open(summary_path, "r", encoding="utf-8") as s_file:
                    summary = s_file.read().strip()

                # Write as JSON line
                json_obj = {
                    "transcript": transcript,
                    "summary": summary
                }
                outfile.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
