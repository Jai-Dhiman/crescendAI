## Capability dossier

- model: `qwen/qwen3.6-35b-a3b`
- routed_providers: []

| Capability | Tier | Primary signal | Value | vs Sonnet | CI | Note |
|---|---|---|---|---|---|---|
| Judgment | mid_tier_with_ceiling_overlap | avg(ASCF, SGD) outcome | 1.57 | -0.22 | [1.46,1.77] |  |
| Taste | at_ceiling | Taste Defensibility (NEW) | 2.98 | n/a | [2.95,3.00] | no baseline anchor |
| Integration | absent_with_mid_overlap | CAP outcome | 1.41 | -0.75 | [1.17,1.65] | inconsistent primary/corroborator |
| Voice | at_ceiling_with_mid_overlap | avg(SPP, ATL, ASM) outcome | 2.70 | -0.19 | [2.64,2.76] |  |
| Vocabulary | at_ceiling | SCML outcome | 3.00 | +0.00 | [3.00,3.00] |  |
| Tool-calling | mid_tier_with_ceiling_overlap | discipline accuracy | 0.76 | n/a | [0.59,0.87] | no baseline anchor |
| Adaptation | mid_tier_with_absent_overlap | Adaptation Specificity (NEW) | 1.65 | n/a | [1.45,1.83] | no baseline anchor |

**Continuation degeneracy rate:** 0.0%; by category: {'clean': 3}

**Over-call rates by negative category:** {'chitchat': 0.0, 'premature': 0.0, 'ambiguous': 0.3333333333333333, 'already_recommended': 0.0, 'out_of_scope': 0.0, 'borderline_wrong_tool': 0.0}
