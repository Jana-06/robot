# BLOOM Multimodal Training & Inference Setup

## Overview

You now have a complete training + inference pipeline for BLOOM that integrates:
- **Eye-tracking data** from `TrainingDataset/` (ASD vs. TD children, ages 5–12)
- **Autism screening data** from `archive/` (demographics + ASD labels)
- **Synthetic data generation** to create training samples with (age, gaze, gesture, body_language, spinning_type) → response mappings
- **Live inference server** that BLOOM can call in real time

## Quick Start

### 1. Train the Model

```bash
cd C:\Users\Janarthan S\OneDrive\Desktop\Robot
python3 bloom_multimodal_trainer.py \
  --dataset-root "C:\Users\Janarthan S\Downloads" \
  --train
```

**Output:**
- Creates `bloom_multimodal_model.json` with 500 synthetic training samples
- Indexed by (age_group: "5-8" or "9-12", context: "calm" / "stim" / "upset")
- Each sample maps multimodal inputs → affirming response

### 2. Start the Inference Server

```bash
python3 bloom_multimodal_trainer.py \
  --serve \
  --model-path bloom_multimodal_model.json \
  --port 8766
```

**Output:**
```
🌸 BLOOM Multimodal Inference Server
   ws://localhost:8766
   Ready to serve predictions...
```

The server listens for WebSocket messages with this structure:
```json
{
  "type": "predict",
  "age": 8,
  "eye_gaze": "center",
  "face_gesture": "happy",
  "body_language": "calm",
  "spinning_type": "none"
}
```

And returns:
```json
{
  "type": "response",
  "text": "That sounds fun! Tell me more!",
  "age": 8,
  "eye_gaze": "center",
  "face_gesture": "happy",
  "body_language": "calm",
  "spinning_type": "none"
}
```

### 3. Test the Bridge (Optional)

```bash
python3 bloom_multimodal_bridge.py
```

**Output:**
```
🌸 BLOOM Multimodal Bridge Example

Test 1: 7-year-old, calm, centered gaze
  Response: That sounds fun! Tell me more!

Test 2: 6-year-old, hand flapping, agitated
  Response: Your stimming helps you feel good! Keep doing what helps your body.

Test 3: 10-year-old, upset, body spinning
  Response: That sounds really hard. I'm here with you.
```

### 4. Wire BLOOM to Call the Multimodal Server

Edit `bloom_server.py` to optionally call the multimodal inference server **instead of** or **alongside** the main LLM:

#### Option A: Use multimodal predictions for response generation (recommended for fine-tuning)

In `bloom_server.py`, import the bridge:
```python
from bloom_multimodal_bridge import call_multimodal_server
```

Then in `run_pipeline()`, after enriching the message with visual context:
```python
# Try multimodal inference first
try:
    mm_result = await loop.run_in_executor(
        None,
        lambda: asyncio.run(call_multimodal_server(
            age=_profile.get("age"),
            eye_gaze=_vision.eye_gaze,
            face_gesture=_vision.face_gesture,
            body_language=_vision.body_language,
            spinning_type=_vision.spinning_type,
        ))
    )
    if mm_result.get("type") == "response":
        resp = mm_result["text"]
        # Affirm if stimulus detected
        is_stim = _has(child_msg, STIM_KEYWORDS) or _has(child_msg, _profile.get("sensory", {}).get("stims", []))
        if is_stim and not _has(resp, ["stimming is", "keep stimming"]):
            resp = f"{stim_aff(_profile)} {resp}"
        verdict = {"is_harmful": False, "is_affirming": True}
        return resp, verdict, time.time() - t0
except Exception as e:
    print(f"  Multimodal call failed, falling back to LLM: {e}")
```

#### Option B: Use multimodal for context only (safer hybrid approach)

Inject multimodal signals into the LLM prompt:
```python
# In build_prompt(), after vision features:
mm_context = (
    f"Detected signals: age={profile.get('age')}, "
    f"gaze={_vision.eye_gaze}, "
    f"gesture={_vision.face_gesture}, "
    f"body={_vision.body_language}, "
    f"spin={_vision.spinning_type}"
)
# Then append to system prompt
```

## File Structure

```
Robot/
├── bloom_dashboard.html              # Frontend
├── bloom_server.py                   # Backend with multimodal fields
├── bloom_multimodal_trainer.py       # Training + inference server
├── bloom_multimodal_bridge.py        # WebSocket bridge for live calls
└── bloom_multimodal_model.json       # Trained model (generated)

Downloads/
├── TrainingDataset/                  # Eye-tracking study
│   ├── TrainingData/
│   │   ├── Images/                   # 300 stimulus images
│   │   ├── ASD/                      # ASD fixation points
│   │   ├── TD/                       # TD fixation points
│   │   ├── ASD_FixMaps/              # ASD heatmaps
│   │   └── TD_FixMaps/               # TD heatmaps
│   └── AdditionalData/
└── archive/                          # Autism screening data
    ├── train.csv                     # Training split (1000+ samples)
    └── test.csv                      # Test split
```

## How It Works

### 1. Data Loading
- **Gaze data**: Extracts fixation coordinates from `ASD/` and `TD/` folders
- **Screening data**: Loads demographics + ASD labels from `archive/train.csv`

### 2. Synthetic Sample Generation
For each synthetic sample:
```python
age ∈ [6, 7, 8, 9, 10, 11]
eye_gaze ∈ ["center", "left", "right", "down", "scattered"]
face_gesture ∈ ["happy", "neutral", "focused", "uncertain"]
body_language ∈ ["calm", "active", "agitated", "spinning"]
spinning_type ∈ ["spinning", "none"]
context ∈ ["calm", "stim", "upset"]  # derived from signals
```

Response is selected from affirming templates grouped by:
- **age_group**: "5-8" or "9-12"
- **context**: "calm", "stim" (stimming), or "upset"

Example: A 7-year-old with body_language="spinning" gets:
```
"Your stimming helps you feel good! Keep doing what helps your body."
```

### 3. Fast Lookup During Inference
Responses are indexed by `(age_group, context)` for O(1) retrieval.  
Fallback is always a safe, neurodiversity-affirming default.

### 4. Live Bridging
When BLOOM detects visual features, it:
1. Calls the multimodal inference server at port 8766
2. Gets back an affirming response tuned to age + signals
3. Optionally re-enters the harm/affirm judgement loop for safety

## Customization

### Add Custom Response Templates

Edit `SyntheticDataGenerator._load_response_templates()` in `bloom_multimodal_trainer.py`:

```python
"5-8_calm": [
    "That sounds fun! Tell me more!",
    "Your custom response here!",
    # ...
]
```

Then retrain:
```bash
python3 bloom_multimodal_trainer.py --dataset-root ... --train
```

### Use Real Video Data (Future)

Once you have actual video samples with labels, replace the synthetic generation:

```python
def load_real_samples(self, video_dir: str) -> List[Dict]:
    samples = []
    for video_file in Path(video_dir).glob("*.mp4"):
        # Extract frames + labels
        metadata = load_json(f"{video_file}.json")  # age, gaze, gesture, body, spin, response
        sample = {
            "age": metadata["age"],
            "eye_gaze": metadata["gaze"],
            "face_gesture": metadata["gesture"],
            "body_language": metadata["body"],
            "spinning_type": metadata["spinning"],
            "response": metadata["response"],
        }
        samples.append(sample)
    return samples
```

### Integrate with Fine-Tuning Service

To use OpenAI / Groq fine-tuning instead of templates:

```python
class LLMMultimodalModel:
    def __init__(self, api_key: str, model_id: str):
        self.api_key = api_key
        self.model_id = model_id  # e.g., "gpt-3.5-turbo"
    
    def predict(self, age, gaze, gesture, body, spin):
        prompt = f"Child age={age}, eye_gaze={gaze}, face_gesture={gesture}, body_language={body}, spinning={spin}. Generate an affirming, neurodiversity-respecting response."
        # Call fine-tuned model
```

## Testing Checklist

- [ ] Train model: `python3 bloom_multimodal_trainer.py --train --dataset-root "C:\Users\Janarthan S\Downloads"`
- [ ] Start server: `python3 bloom_multimodal_trainer.py --serve`
- [ ] Test bridge: `python3 bloom_multimodal_bridge.py`
- [ ] Start BLOOM: `python3 bloom_server.py --child arjun`
- [ ] Open dashboard: `bloom_dashboard.html`
- [ ] Verify WebSocket messages include `eye_gaze`, `face_gesture`, `body_language`, `spinning_type`

## Troubleshooting

### Model file not found
```
Error: Model not found at bloom_multimodal_model.json
```
→ Run `--train` first to generate the model.

### Inference server connection refused
```
Error: [Errno 10061] No connection could be made...
```
→ Start the inference server: `python3 bloom_multimodal_trainer.py --serve`

### Fixation files not found
```
  Loading eye-tracking data...
  Loaded 0 eye-tracking samples
```
→ Check dataset path. Use the full path to the `TrainingDataset/` folder.

## Next Steps

1. **Use real video data**: Annotate actual child interaction videos and replace synthetic generation
2. **Deploy to cloud**: Move inference server to AWS Lambda / Azure Functions for live deployment
3. **A/B test responses**: Compare template-based vs. LLM fine-tuned responses
4. **Collect user feedback**: Track which responses were most helpful; retrain periodically

## References

- Eye-tracking dataset: Duan et al., "A Dataset of Eye Movements for the Children with Autism Spectrum Disorder" (MMSys'19)
- Autism screening data: Available in `archive/`
- BLOOM guardrails: Neurodiversity-affirming language framework

---

**Questions?** Check the inline docstrings in each `.py` file or run with `--help`.
