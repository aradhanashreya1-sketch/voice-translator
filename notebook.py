import torch
import whisper
import gradio as gr
import os
import re
import subprocess
import soundfile as sf
import numpy as np
import requests
from deep_translator import GoogleTranslator
from TTS.api import TTS

HOSTINGER_URL = "http://82.112.226.181:8080/api/kaggle-url"

def update_hostinger(gradio_url):
    try:
        requests.post(HOSTINGER_URL, json={"url": gradio_url}, timeout=5)
        print(f"✅ Hostinger updated: {gradio_url}")
    except Exception as e:
        print(f"⚠️ Hostinger update failed: {e}")

print("⏳ 1/2: Whisper load ho raha hai...")
whisper_model = whisper.load_model("base")
print("✅ Whisper ready!")

print("⏳ 2/2: XTTS v2 load ho raha hai...")
xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print("✅ XTTS ready!")

INPUT_LANGUAGES = {
    "Auto Detect 🔍": None,
    "Hindi 🇮🇳": "hi",
    "English 🇬🇧": "en",
    "Urdu 🇵🇰": "ur",
    "Bengali 🇧🇩": "bn",
    "Tamil 🇮🇳": "ta",
    "Telugu 🇮🇳": "te",
    "Marathi 🇮🇳": "mr",
    "Gujarati 🇮🇳": "gu",
    "Punjabi 🇮🇳": "pa",
    "Kannada 🇮🇳": "kn",
    "Malayalam 🇮🇳": "ml",
    "Arabic 🇸🇦": "ar",
    "Chinese 🇨🇳": "zh",
    "Japanese 🇯🇵": "ja",
    "Korean 🇰🇷": "ko",
    "Spanish 🇪🇸": "es",
    "French 🇫🇷": "fr",
    "German 🇩🇪": "de",
    "Russian 🇷🇺": "ru",
}

OUTPUT_LANGUAGES = {
    "Hindi 🇮🇳": "hi",
    "English 🇬🇧": "en",
    "Spanish 🇪🇸": "es",
    "French 🇫🇷": "fr",
    "German 🇩🇪": "de",
    "Italian 🇮🇹": "it",
    "Portuguese 🇧🇷": "pt",
    "Polish 🇵🇱": "pl",
    "Dutch 🇳🇱": "nl",
    "Czech 🇨🇿": "cs",
    "Hungarian 🇭🇺": "hu",
    "Turkish 🇹🇷": "tr",
    "Russian 🇷🇺": "ru",
    "Arabic 🇸🇦": "ar",
    "Chinese 🇨🇳": "zh-cn",
    "Japanese 🇯🇵": "ja",
    "Korean 🇰🇷": "ko",
}

def split_text(text, max_chars=200):
    sentences = re.split(r'(?<=[.!?।,;:])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(s) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            words = s.split()
            temp = ""
            for w in words:
                if len(temp) + len(w) < max_chars:
                    temp += " " + w
                else:
                    if temp:
                        chunks.append(temp.strip())
                    temp = w
            if temp:
                chunks.append(temp.strip())
        elif len(current) + len(s) < max_chars:
            current += " " + s
        else:
            if current:
                chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c.strip()] or [text[:200]]

def prepare_speaker(audio_path):
    try:
        out = "/kaggle/working/speaker.wav"
        subprocess.run([
            "ffmpeg", "-i", audio_path,
            "-ss", "0", "-t", "30",
            "-ar", "22050", "-ac", "1",
            "-af", "highpass=f=200,lowpass=f=3000",
            "-y", out
        ], capture_output=True)
        return out if os.path.exists(out) else audio_path
    except:
        return audio_path

def join_chunks(chunk_files, output_path):
    if len(chunk_files) == 1:
        os.rename(chunk_files[0], output_path)
    else:
        list_file = "/kaggle/working/list.txt"
        with open(list_file, "w") as f:
            for cf in chunk_files:
                f.write(f"file '{cf}'\n")
        subprocess.run([
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", list_file, "-c", "copy",
            "-y", output_path
        ], capture_output=True)
        for cf in chunk_files:
            try: os.remove(cf)
            except: pass

def transcribe_audio(audio_path, manual_text, input_language_name):
    if manual_text and manual_text.strip():
        return manual_text.strip(), "✅ Manual text use kiya!"
    if audio_path is None:
        return "", "❌ Audio upload karo ya text type karo!"
    try:
        lang_code = INPUT_LANGUAGES[input_language_name]
        result = whisper_model.transcribe(
            audio_path, fp16=True, verbose=False,
            language=lang_code, task="transcribe",
            initial_prompt="हिंदी में लिखें।" if lang_code == "hi" else None
        )
        text = result["text"].strip()
        detected = result["language"]
        lang_info = f"Auto Detected: {detected.upper()}" if lang_code is None else f"Language: {input_language_name}"
        return text, f"✅ Transcribed!\n🌐 {lang_info}\n✏️ Edit kar sakte ho!"
    except Exception as e:
        return "", f"❌ Error: {str(e)}"

def translate_text(original_text, target_language_name):
    if not original_text or not original_text.strip():
        return "", "❌ Pehle Step 1 karo!"
    try:
        lang_code = OUTPUT_LANGUAGES[target_language_name]
        translated = GoogleTranslator(source="auto", target=lang_code).translate(original_text)
        return translated, "✅ Translation ho gayi!\n✏️ Verify karo — phir Generate dabao!"
    except Exception as e:
        return "", f"❌ Error: {str(e)}"

def generate_audio(audio_path, final_text, target_language_name):
    if not final_text or not final_text.strip():
        return None, "❌ Text empty hai!"
    if audio_path is None:
        return None, "❌ Original audio chahiye!"
    try:
        lang_code = OUTPUT_LANGUAGES[target_language_name]
        speaker = prepare_speaker(audio_path)
        output_path = "/kaggle/working/final_output.wav"
        chunks = split_text(final_text)
        chunk_files = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            chunk_path = f"/kaggle/working/chunk_{i}.wav"
            print(f"🎵 Chunk {i+1}/{len(chunks)}: {chunk[:40]}...")
            xtts_model.tts_to_file(
                text=chunk, speaker_wav=speaker,
                language=lang_code, file_path=chunk_path,
                temperature=0.65, repetition_penalty=5.0,
                top_k=50, top_p=0.85, speed=1.0,
            )
            chunk_files.append(chunk_path)
        join_chunks(chunk_files, output_path)
        return output_path, f"✅ Ready!\n\n🌍 Language: {target_language_name}\n📝 Text:\n{final_text}"
    except Exception as e:
        import traceback
        return None, f"❌ Error:\n{str(e)}\n\n{traceback.format_exc()}"

with gr.Blocks(title="🎙️ Voice Translator Pro", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎙️ Voice Translator Pro")
    gr.Markdown("**XTTS v2 — 17 Languages | 3 Simple Steps**")

    with gr.Group():
        gr.Markdown("## 1️⃣ Input Do")
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(label="🎤 Audio Upload Karo", type="filepath", sources=["upload", "microphone"])
            with gr.Column():
                manual_text_input = gr.Textbox(label="✍️ YA Text Direct Type Karo", placeholder="Agar audio nahi hai toh yahan type karo...", lines=5)
        input_language_select = gr.Dropdown(choices=list(INPUT_LANGUAGES.keys()), value="Auto Detect 🔍", label="🎤 Input Audio Language")
        transcribe_btn = gr.Button("📝 Step 1: Transcribe / Text Use Karo", variant="secondary", size="lg")
        transcribe_status = gr.Textbox(label="Status", lines=3, interactive=False)

    with gr.Group():
        gr.Markdown("## 2️⃣ Edit aur Translate Karo")
        original_text_box = gr.Textbox(label="✏️ Original Text — Edit Kar Sakte Ho", placeholder="Step 1 ke baad yahan aayega...", lines=4, interactive=True)
        with gr.Row():
            output_language_select = gr.Dropdown(choices=list(OUTPUT_LANGUAGES.keys()), value="English 🇬🇧", label="🌍 Output Language")
            translate_btn = gr.Button("🌍 Step 2: Translate Karo", variant="secondary", size="lg")
        translate_status = gr.Textbox(label="Status", lines=2, interactive=False)

    with gr.Group():
        gr.Markdown("## 3️⃣ Verify Karo aur Audio G
