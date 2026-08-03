# model/src/follower_eval/
"""Real-audio score-follower eval (issue #133): the SOURCE-OF-TRUTH eval.

Unlike ``follower_bench`` (synthetic ASAP MIDI + spliced pathologies, to be
pruned once this is trusted), everything here runs on REAL YouTube -> AMT
transcribed practice recordings -- phone-quality audio, real amateur playing,
real transcription noise. No synthetic clips, no augmentation.

Two metric tracks:
  * proxy (this module, ``realaudio``): anchor-free structural signals over ALL
    transcribed clips -- coverage, score-span traversed, monotonicity-where-
    expected, confidence + its calibration. Measures "does the follower behave
    sanely at scale"; needs no labels.
  * gold (S3, layered on later): human-tapped bar onsets give an independent
    (audio_sec -> score bar) reference so cursor ACCURACY is a real, non-circular
    number on a labeled subset.
"""
