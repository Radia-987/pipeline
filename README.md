# Emotion Recognition Pipeline 🎭

Therapeutic conversation analysis system with AI-powered speaker diarization, emotion detection, and GPT-4o enhanced speaker identification for Urdu and English conversations.

## Features ✨

- **Speaker Diarization**: PyAnnote acoustic diarization + GPT-4o context-aware correction
- **Smart Speaker Identification**: Automatically identifies THERAPIST vs PATIENT
- **Emotion Recognition**: 
  - Audio emotion detection (Fine-tuned Wav2Vec2)
  - Text emotion analysis (DistilRoBERTa)
- **Multilingual Support**: Urdu and English transcription with Whisper
- **Translation**: GPT-4o powered Urdu → English translation
- **Interactive Dashboard**: Streamlit-based web interface

---

## Prerequisites 📋

- Python 3.10 or 3.11
- OpenAI API key (for Whisper & GPT-4o)
- HuggingFace token (for PyAnnote diarization)
- Fine-tuned Wav2Vec2 model (placed at `C:\Users\Radia\Desktop\model\fine_tuned_wav2vec2`)

---

## Installation 🚀

### Step 1: Clone the repository
```powershell
git clone https://github.com/Radia-987/pipeline.git
cd pipeline
```

### Step 2: Create virtual environment
```powershell
python -m venv venv
```

### Step 3: Activate virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4: Install dependencies
```powershell
pip install -r er_requirements.txt
```

### Step 5: Set up environment variables

Create a `.env` file in the project root:
```powershell
New-Item -Path .env -ItemType File
```

Open `.env` and add your API keys:
```
OPENAI_API_KEY=sk-proj-your_openai_key_here
HF_TOKEN=hf_your_huggingface_token_here
```

**How to get these keys:**

**OpenAI API Key:**
1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Copy and paste into `.env`

**HuggingFace Token:**
1. Go to https://huggingface.co/settings/tokens
2. Create new token (read access)
3. Accept PyAnnote terms: https://huggingface.co/pyannote/speaker-diarization-3.1
4. Copy and paste into `.env`

---

## Running the Application ▶️

```powershell
streamlit run er_pipeline.py
```

The app will open in your browser at `http://localhost:8501`

---

## Usage 📖

1. **Upload Audio**: Upload WAV, MP3, M4A, or FLAC file
2. **Configure Settings** (optional):
   - Expected speakers (set to 2 for therapy sessions)
   - Aggregation mode for emotion plotting
   - Confidence filters
3. **Click "Analyze"**: The system will:
   - Diarize speakers using PyAnnote
   - Transcribe with Whisper (auto-detects Urdu/English)
   - Correct speaker labels with GPT-4o (THERAPIST/PATIENT)
   - Detect emotions (audio + text)
   - Translate to English
   - Generate visualizations
4. **View Results**:
   - PyAnnote diarization (raw acoustic)
   - GPT-corrected transcript (THERAPIST/PATIENT)
   - Patient emotion timeline charts
   - Sentence-level emotion analysis
   - Downloadable reports

---

## Project Structure 📁

```
aun-pipeline/
├── er_pipeline.py          # Main Streamlit application
├── er_requirements.txt     # Python dependencies
├── .env                    # API keys (create this)
├── .gitignore             # Git ignore rules
├── README.md              # This file
└── venv/                  # Virtual environment (created by you)
```

---

## Sidebar Controls 🎛️

### Utterance Merging
- **Max pause**: How long a silence can be before splitting utterances (0.2-2.0s)

### Plot Controls
- **Aggregation mode**: 
  - Per utterance (most detailed)
  - Time window (smoothed by time chunks)
  - Change-only (only when emotion changes)
- **Window size**: For time window mode (10-120s)
- **Max points**: Limit chart complexity (50-1000)
- **Min confidence**: Filter low-confidence predictions (0.0-1.0)

### Diarization
- **Expected speakers**: Set to 2 for therapy sessions (0 = auto-detect)

---

## Technologies Used 🛠️

- **Frontend**: Streamlit
- **Transcription**: OpenAI Whisper-1
- **Translation**: OpenAI GPT-4o
- **Diarization**: PyAnnote speaker-diarization-3.1
- **Audio Emotion**: Fine-tuned Wav2Vec2
- **Text Emotion**: j-hartmann/emotion-english-distilroberta-base
- **Audio Processing**: librosa, pydub, torchaudio
- **Visualization**: Plotly, Altair

---

## Troubleshooting 🔧

### "Missing API keys" error
- Ensure `.env` file exists in project root
- Check that keys are correctly formatted (no extra spaces)
- Verify keys are valid and active

### Audio processing errors
- Ensure audio file is not corrupted
- Try converting to WAV format first
- Check if audio is at least 1 second long

### Diarization fails
- Verify HuggingFace token has access to PyAnnote models
- Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
- Set "Expected speakers" to 2 for better results

### Slow performance
- Use shorter audio clips (under 5 minutes recommended)
- Increase "Max points to display" slider value
- Use "Time window" aggregation mode

---

## License 📄

This project is for research and educational purposes.

---

## Contact 📧

For issues or questions, please open an issue on GitHub.

---

## Acknowledgments 🙏

- OpenAI for Whisper and GPT-4o
- PyAnnote team for speaker diarization
- HuggingFace for model hosting
- Streamlit for the web framework
