import asyncio
import edge_tts
import nltk
import os
import warnings
from flask import Flask, render_template, request, jsonify
from transformers import pipeline, logging
from pydub import AudioSegment
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab', quiet=True)
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

app = Flask(__name__)
model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

logic_map = {
    'joy':     {'rate': '+15%', 'pitch': '+8Hz',  'volume': '+10%'},
    'anger':   {'rate': '+10%', 'pitch': '+3Hz',  'volume': '+15%'},
    'fear':    {'rate': '+10%', 'pitch': '+5Hz',  'volume': '+0%'},
    'sadness': {'rate': '-15%', 'pitch': '-8Hz',  'volume': '-15%'},
    'disgust': {'rate': '-8%',  'pitch': '-3Hz',  'volume': '-8%'},
    'surprise':{'rate': '+12%', 'pitch': '+10Hz', 'volume': '+12%'},
    'neutral': {'rate': '+0%',  'pitch': '+0Hz',  'volume': '+0%'},
}

def apply_intensity(val_str, score):
    if 'Hz' in val_str: unit = 'Hz'
    else: unit = '%'
    value = float(val_str[1:].replace('%','').replace('Hz',''))
    scaled = value * score
    return f"{val_str[0]}{int(scaled)}{unit}"

async def generate_audio(sentence, params, filename):
    communicate = edge_tts.Communicate(
        text=sentence,
        voice="en-US-JennyNeural",
        rate=params['rate'],
        pitch=params['pitch'],
        volume=params['volume']
    )
    await communicate.save(filename)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    data = request.json
    text = data["text"]
    sentences = sent_tokenize(text)
    parts = []
    results = []

    for i, sentence in enumerate(sentences):
        result = model(sentence)
        emotion = result[0]['label']
        score = result[0]['score']

        parameters = logic_map.get(emotion, logic_map['neutral'])
        parameters = {k: apply_intensity(v, score) for k, v in parameters.items()}

        filename = f"static/part{i}.mp3"
        asyncio.run(generate_audio(sentence, parameters, filename))
        parts.append(filename)
        results.append({'sentence': sentence, 'emotion': emotion, 'score': round(score, 2)})

    silence = AudioSegment.silent(duration=173)
    full = AudioSegment.empty()
    for part in parts:
        full += AudioSegment.from_mp3(part) + silence

    full.export("static/output.mp3", format="mp3")

    for part in parts:
        os.remove(part)

    return jsonify({"status": "success", "audio": "/static/output.mp3", "results": results})

if __name__ == "__main__":
    app.run(debug=True)