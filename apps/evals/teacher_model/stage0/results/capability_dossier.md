## Capability dossier

- model: `qwen/qwen3.6-35b-a3b`
- routed_providers: []

| Capability | Tier | Primary signal | Value | vs Sonnet | CI | Note |
|---|---|---|---|---|---|---|
| Judgment | mid_tier_with_ceiling_overlap | avg(ASCF, SGD) outcome | 1.53 | -0.26 | [1.45,1.77] |  |
| Taste | at_ceiling | Taste Defensibility (NEW) | 2.98 | n/a | [2.95,3.00] | no baseline anchor |
| Integration | mid_tier_with_absent_overlap | CAP outcome | 1.42 | -0.75 | [1.20,1.66] |  |
| Voice | at_ceiling_with_mid_overlap | avg(SPP, ATL, ASM) outcome | 2.68 | -0.22 | [2.62,2.73] |  |
| Vocabulary | at_ceiling | SCML outcome | 3.00 | +0.00 | [3.00,3.00] |  |
| Tool-calling | mid_tier_with_ceiling_overlap | discipline accuracy | 0.75 | n/a | [0.60,0.86] | no baseline anchor |
| Adaptation | mid_tier_with_absent_overlap | Adaptation Specificity (NEW) | 1.57 | n/a | [1.39,1.75] | no baseline anchor |

**Continuation degeneracy rate:** 0.0%; by category: {'clean': 12}

**Over-call rates by negative category:** {'chitchat': 0.0, 'premature': 0.0, 'ambiguous': 0.3333333333333333, 'already_recommended': 0.3333333333333333, 'out_of_scope': 0.0, 'borderline_wrong_tool': 0.0}
