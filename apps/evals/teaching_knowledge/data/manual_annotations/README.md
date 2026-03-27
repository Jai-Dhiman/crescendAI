# Manual Annotations for Extraction Calibration

Annotate 20 transcripts (10 masterclass, 10 lesson) using the 4-field schema.
Save each as a JSON file named {video_id}.json with this structure:

```json
[
  {
    "what_teacher_said": "verbatim or close paraphrase",
    "dimension_focus": "dynamics|timing|pedaling|articulation|phrasing|interpretation|general",
    "student_skill_estimate": "beginner|early_intermediate|intermediate|advanced|professional",
    "feedback_type": "corrective|encouraging|modeling|guided_discovery|scaffolding|motivational"
  }
]
```

Target: 20 transcripts annotated by founder (Jai).
