# BLOOM Multimodal System — Architecture & Files

## What Was Built

You now have a complete **multimodal training + inference system** that:

1. **Loads both datasets** you provided
   - Eye-tracking data (ASD vs. TD children, ages 5–12)
   - Autism screening questionnaire data (demographics + ASD labels)

2. **Generates synthetic training samples** mapping:
   - Age (6–11)
   - Eye gaze (center, left, right, down, scattered)
   - Face gesture (happy, neutral, focused, uncertain)
   - Body language (calm, active, agitated, spinning)
   - Spinning type (spinning, none)
   - **→ Affirming response text**

3. **Trains a lightweight response model** indexed by (age_group, context)

4. **Serves predictions live** over WebSocket on port 8766

5. **Integrates with BLOOM** to use visual signals for response generation

## New Files

```
Robot/
├── bloom_multimodal_trainer.py          # Main training + inference server
├── bloom_multimodal_bridge.py           # WebSocket bridge for testing
├── bloom_multimodal_integration.py      # 3 integration patterns for BLOOM
├── MULTIMODAL_SETUP.md                  # Comprehensive setup guide
├── requirements.txt                      # Python dependencies
├── setup.bat                             # Windows batch setup script
├── setup.ps1                             # PowerShell setup script
└── ARCHITECTURE.md                       # This file

Modified:
├── bloom_server.py                      # Added multimodal fields + set_visual_features hook
└── bloom_dashboard.html                 # Dashboard receives new frame fields
```

## Quick Start (3 Commands)

### Windows (PowerShell)
```powershell
cd C:\Users\Janarthan S\OneDrive\Desktop\Robot
.\setup.ps1
```

### Windows (CMD)
```cmd
cd C:\Users\Janarthan S\OneDrive\Desktop\Robot
setup.bat
```

### Manual (any OS)
```bash
pip install -r requirements.txt
python bloom_multimodal_trainer.py --dataset-root "C:\Users\Janarthan S\Downloads" --train
```

## How the System Works

### Phase 1: Training (One-Time)

```
bloom_multimodal_trainer.py --train
    ↓
    ├─→ Load TrainingDataset/
    │       ├─ ASD fixation points (eyes)
    │       ├─ TD fixation points
    │       └─ 300 stimulus images
    │
    ├─→ Load archive/
    │       ├─ Autism screening questionnaire
    │       └─ Age, gender, ASD labels (0/1)
    │
    ├─→ Generate 500 synthetic samples
    │       ├─ age, eye_gaze, face_gesture, body_language, spinning_type
    │       └─ → affirming response (from templates)
    │
    └─→ Save bloom_multimodal_model.json
            ├─ 500 training samples
            └─ Response index: {(age_group, context): [responses]}
```

### Phase 2: Inference (Live)

```
bloom_multimodal_trainer.py --serve
    ↓
    Load bloom_multimodal_model.json
    ↓
    Listen on ws://localhost:8766
    ↓
    For each prediction request:
        age, eye_gaze, face_gesture, body_language, spinning_type
        ↓
        Determine context (calm / stim / upset)
        ↓
        Lookup response: response_index[(age_group, context)]
        ↓
        Return response text
```

### Phase 3: Integration with BLOOM (Optional)

```
bloom_server.py --child arjun
    ↓
    Capture visual features (face, emotion, action, spin)
    ↓
    User sends message
    ↓
    Call multimodal inference server (port 8766)
        send: age, eye_gaze, face_gesture, body_language, spinning_type
        recv: affirming response
    ↓
    Option A: Use response directly
    Option B: Use as context for main LLM
    Option C: Run through harm/affirm judges
    ↓
    Send response to child via TTS
```

## Integration Options

See `bloom_multimodal_integration.py` for 3 patterns:

### Pattern 1: Direct Replacement (Aggressive)
```python
# Skip LLM, use multimodal response directly
response = await get_multimodal_response(age, gaze, gesture, body, spin)
```
**Pros:** Fast, predictable  
**Cons:** No LLM fallback if signals ambiguous

### Pattern 2: Context Enrichment (Safe)
```python
# Inject multimodal signals into LLM prompt
system_prompt += f"[Visual: gaze={gaze}, gesture={gesture}, body={body}]"
response = await llm(system_prompt, message)
```
**Pros:** LLM can reason over signals  
**Cons:** Slower, more API calls

### Pattern 3: Hybrid (Balanced)
```python
# Use multimodal to classify emotion, then LLM generates with that hint
state = infer_state_from_multimodal(...)  # "upset" / "stim" / "calm"
response = await llm_with_state_hint(message, state)
```
**Pros:** Fast classification + flexible generation  
**Cons:** Requires tuning both models

## Data Flow

### Bloom Dashboard → Bloom Server → Multimodal Server

```
Browser                  BLOOM Server              Multimodal Server
  ↓                           ↓                          ↓
Camera frame  →  video_frame msg  →  _vision.process()
                                          ↓
                                   eye_gaze
                                   face_gesture
                                   body_language
                                   spinning_type
                                          ↓
User message  →  text_msg         →  handle_msg()
                                          ↓
                                   set_visual_features
                                   (optional, from ML)
                                          ↓
                              call multimodal server
                                          ↓
                            ws.send({type: "predict"})
                                          ↓
                        Multimodal Server receives
                        ↓
                        Lookup response
                        ↓
                        ws.send({type: "response", text: "..."})
                                          ↓
                         BLOOM processes response
                         (harm check, affirm check)
                                          ↓
                         TTS + dashboard update
```

## File Responsibilities

| File | Role |
|------|------|
| `bloom_multimodal_trainer.py` | DatasetLoader, SyntheticDataGenerator, MultimodalResponseModel, Inference server |
| `bloom_multimodal_bridge.py` | WebSocket client for calling inference server |
| `bloom_multimodal_integration.py` | Integration patterns + standalone test |
| `bloom_server.py` | (Modified) Stores multimodal fields, sends over websocket, handles set_visual_features |
| `bloom_dashboard.html` | (Unchanged) Receives multimodal frame data in onFrame() |
| `MULTIMODAL_SETUP.md` | Complete setup + customization guide |
| `requirements.txt` | Python dependencies |
| `setup.ps1` / `setup.bat` | One-command setup |

## Key Signals in Pipeline

### Child Profile (bloom_server.py)
```python
profile["vision_features"] = {
    "eye_gaze": "center|left|right|down|scattered|unknown",
    "face_gesture": "happy|neutral|focused|uncertain|sad|afraid|angry|unknown",
    "body_language": "calm|active|agitated|spinning|unknown",
    "spinning_type": "spinning|none",
}
profile["age"] = 6  # Updated from set_visual_features msg
```

### Vision Detector Output (multimodal server)
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

### Response
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

## Customization Points

1. **Response templates** → Edit `SyntheticDataGenerator._load_response_templates()` in trainer
2. **Age groups** → Modify age bins (currently 5-8, 9-12)
3. **Gaze categories** → Add more fine-grained categories (e.g., "upper-left", "lower-right")
4. **Context detection** → Tune rules in `_refresh_visual_features()` (bloom_server.py)
5. **Real video data** → Replace synthetic generation with video annotation loader

## Testing Checklist

- [ ] `pip install -r requirements.txt`
- [ ] `python bloom_multimodal_trainer.py --dataset-root "C:\Users\Janarthan S\Downloads" --train`
- [ ] `python bloom_multimodal_trainer.py --serve` (in one terminal)
- [ ] `python bloom_multimodal_bridge.py` (in another terminal — should see responses)
- [ ] `python bloom_server.py --child arjun` (in another terminal)
- [ ] Open `bloom_dashboard.html`
- [ ] Check browser console for WebSocket messages with `eye_gaze`, `face_gesture`, `body_language`, `spinning_type`

## Next Steps (Priority Order)

1. **Annotate real video data** (age, gaze, gesture, body, spin, response)
2. **Collect user feedback** from child interactions
3. **Fine-tune with real data** (replace synthetic generation)
4. **Deploy inference server** to cloud (AWS Lambda, Azure Functions)
5. **A/B test response styles** (template vs. LLM vs. hybrid)
6. **Add emotion classifier** (happy, sad, angry, calm, excited)
7. **Track long-term patterns** (improving over sessions)

---

**Questions?** See `MULTIMODAL_SETUP.md` for troubleshooting and detailed examples.
