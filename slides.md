# Slide 1 - Problem
- Autistic children are often forced into compliance-oriented interaction systems.
- BLOOM centers neurodiversity-affirming support, not normalization.
- Goal: build a child-AI companion that respects stimming, communication differences, and sensory regulation.

# Slide 2 - System Architecture
- Inputs: video frames, eye gaze, face gesture, body language, age, spinning/stim behavior.
- Pipeline: multimodal inference server + BLOOM websocket companion server + live dashboard.
- Outputs: affirming responses, engagement score/trend, stim phase, RL adaptation signals.

# Slide 3 - Co-Adaptive RL + Multimodal Pipeline
- Engagement predictor computes 0-100 score from multimodal signals with rolling trend.
- Online RL uses next-frame implicit reward from face/body/gaze transitions.
- Epsilon-greedy response selector adapts per (age_group, context) bucket.
- Current adaptation status: 0 updates across 0 buckets.

# Slide 4 - Live Demo Results
- Total interactions: 0
- Average engagement: 0.0
- Dominant context: calm
- Reward outcomes: +0 / -0 / pending 0

# Slide 5 - NeuroDesign Alignment
- Human Brain <-> Robot Brain: policy adapts from implicit multimodal feedback.
- Human Brain <-> Robot Body: visual signals shape real-time verbal/TTS behavior.
- Robot Brain <-> Human Body: supportive responses aim to regulate arousal, preserve autonomy.
- Human Body <-> Robot Body: stim-phase detection drives respectful co-regulation.
