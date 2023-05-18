import whisper

model = whisper.load_model("base")
result = model.transcribe("Tino.mp3")
print(result["text"])
