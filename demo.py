from faster_whisper import WhisperModel

# download the video and extract audio using yt-dlp
# yt-dlp -x --audio-format m4a "https://www.youtube.com/watch?v=VIDEO_ID"
# yt-dlp -x --audio-format m4a --download-sections "*00:00:00-00:00:30" "https://www.youtube.com/watch?v=VIDEO_ID"


model = WhisperModel("large-v2", device="auto", compute_type="auto")

segments, info = model.transcribe("VIDEO_ID.m4a")

times = []
tagalog_text = ""
for segment in segments:
    tagalog_text += segment.text + " "
    start_time = segment.start
    end_time = segment.end
    times.append((start_time, end_time))
    # print(f"Start: {start_time:.2f}s, End: {end_time:.2f}s, Text: {segment.text}")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipelines

model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Set languages (Languages in FLORES-200)
src_lang = "tgl_Latn"     # Tagalog
tgt_lang = "zho_Hant"     # Chinese (Traditional)

translator = pipelines.TranslationPipeline(model=model, tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang)

output = translator(tagalog_text)
translated_text = output[0]['translation_text']

# Save the translated text to a file with the timestamp
with open("translated_text.txt", "w", encoding="utf-8") as f:
    for start, end in times:
        f.write(f"[{start:.2f}s - {end:.2f}s] {translated_text}\n")
print("Translated text saved to translated_text.txt")
