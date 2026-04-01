import asyncio
import edge_tts
import nltk
import os
import time
import warnings
from transformers import pipeline
from transformers import logging
from pydub import AudioSegment
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab', quiet=True)
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

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

#based on score we change values of parameter
def apply_intensity(val_str, score):
    if 'Hz' in val_str: unit = 'Hz'
    else: unit = '%'

    value = float(val_str[1:].replace('%','').replace('Hz',''))
    scaled = value*score

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

while True:
    text = input("\nEnter your text('quit' to exit):")
    if text.lower() == 'quit': break;

    sentences = sent_tokenize(text)
    parts = []

    for i, sentence in enumerate(sentences):
        result = model(sentence)
        emotion = result[0]['label']
        score = result[0]['score']
        print(f"Sentence {i+1}: '{sentence}' → {emotion} ({score:.2f})")

        parameters = logic_map.get(emotion, logic_map['neutral'])
        parameters = {k: apply_intensity(v, score) for k,v in parameters.items()}

        filename = f"part{i}.mp3"
        asyncio.run(generate_audio(sentence, parameters, filename))
        parts.append(filename)

    silence = AudioSegment.silent(duration = 184)
    full= AudioSegment.empty()
    for part in parts:
        full += AudioSegment.from_mp3(part) + silence

    full.export("output.mp3", format="mp3")
    print("Saved output.mp3")

    for part in parts:
        os.remove(part)

