# Empathy Engine

An AI-powered Text-to-Speech service that dynamically modulates vocal characteristics based on the detected emotion of the input text. The Empathy Engine analyzes the emotional content of each sentence and adjusts rate, pitch, and volume to produce expressive, human-like speech.

---

## Features

- **Sentence-level emotion detection** — each sentence is analyzed independently, allowing mixed emotions within a single paragraph
- **7 emotion categories** — joy, anger, fear, sadness, disgust, surprise, neutral
- **Intensity scaling** — vocal parameters are scaled by the model's confidence score. Higher confidence in an emotion produces more pronounced vocal modulation, resulting in more expressive output.
- **Research-backed mappings** — emotion-to-voice mappings are derived from Pol van Rijn & Pauline Larrouy-Maestri (2023), *Nature Human Behaviour*
- **Web interface** — clean Flask-based UI with real-time emotion breakdown and embedded audio player

---

## Tech Stack

- **Sentence Tokenization**: `nltk`
- **Emotion Detection**: `j-hartmann/emotion-english-distilroberta-base` (Hugging Face Transformers)
- **Text-to-Speech**: Microsoft Edge TTS (`edge-tts`) via `en-US-JennyNeural` voice
- **Audio Processing**: `pydub` for sentence stitching
- **Web Framework**: Flask

---

## Setup

### Prerequisites
- Python 3.11
- ffmpeg (`brew install ffmpeg` on Mac)

### Installation
```bash
git clone https://github.com/Shashankk002/empathy-engine.git
cd empathy-engine

python3.11 -m venv venv
source venv/bin/activate

pip install flask transformers torch edge-tts pydub nltk
```

---

## Running the App
```bash
source venv/bin/activate
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

---

## Design Choices

### Emotion Detection Model
We use `j-hartmann/emotion-english-distilroberta-base`, a RoBERTa-based transformer fine-tuned for 7-class emotion classification. This was chosen over simpler lexicon-based approaches (TextBlob, VADER) because:
- It captures context, not just individual words
- 7 emotion categories fulfill the granular emotions bonus objective
- It is the most downloaded emotion classification model on Hugging Face for English text

### Emotion-to-Voice Mapping
Vocal parameters (rate, pitch, volume) are mapped to each emotion based on empirical findings from:

> van Rijn, P. & Larrouy-Maestri, P. (2023). *Modelling individual and cross-cultural variation in the mapping of emotions to speech prosody*. Nature Human Behaviour, 7, 386–396. https://doi.org/10.1038/s41562-022-01505-5

The mappings are derived from Figure 2a of the paper, which reports global acoustic coefficients across corpora for RC2 (Loudness → volume), RC3 (Pitch and formants → pitch), and RC4 (Rhythm and tempo → rate):

| Emotion  | Rate  | Pitch  | Volume |
|----------|-------|--------|--------|
| Joy      | +15%  | +8Hz   | +10%   |
| Anger    | +10%  | +3Hz   | +15%   |
| Fear     | +10%  | +5Hz   | +0%    |
| Sadness  | -15%  | -8Hz   | -15%   |
| Disgust  | -8%   | -3Hz   | -8%    |
| Surprise | +12%  | +10Hz  | +12%   |
| Neutral  | +0%   | +0Hz   | +0%    |

### Intensity Scaling
Each parameter is scaled by the model's confidence score. A sentence classified as joy with 95% confidence will sound more expressive than one classified with 55% confidence. This implements the "Intensity Scaling" bonus objective.

### Sentence-Level Processing
Input text is split into sentences using NLTK's `sent_tokenize`. Each sentence is processed independently and the resulting audio clips are stitched together with a 234ms silence gap using pydub.

---

## Limitations & Future Work

- Voice consistency across sentences is limited by edge-tts generating each sentence independently
- Pitch manipulation is constrained to edge-tts native parameters; full acoustic control (shimmer, jitter, formants) would require a vocoder like Praat or a neural TTS like ElevenLabs
- Model accuracy drops on very short or ambiguous sentences
- The model's confidence score does not always reflect linguistic emphasis. For example, "This is the BEST DAY EVER" may score lower confidence than "This is good" for the same emotion, since the model reads semantic content rather than typographic emphasis. A heuristic caps/exclamation boost is applied to partially compensate for this.

---

## References

- van Rijn, P. & Larrouy-Maestri, P. (2023). Modelling individual and cross-cultural variation in the mapping of emotions to speech prosody. *Nature Human Behaviour*, 7, 386–396. https://doi.org/10.1038/s41562-022-01505-5
- Hartmann, J. et al. `j-hartmann/emotion-english-distilroberta-base`. Hugging Face. https://huggingface.co/j-hartmann/emotion-english-distilroberta-base