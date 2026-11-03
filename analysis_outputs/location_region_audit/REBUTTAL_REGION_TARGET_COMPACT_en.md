### Independent evaluation of non-object/region-like targets

We follow the existing taxonomy of 37 non-object/region-like anchors among 199 anchors, without modifying the annotations or adding region boundaries. The 4,000 interactions are grouped by their GT sets into object-only, non-object-only, and mixed interactions; mixed interactions retain all GT targets and are reported separately.

| Interaction group | # samples | Anchor Micro-F1 | 2D Joint-F1@200 |
|---|---:|---:|---:|
| Object-only | 2,628 | 0.6786 (best open-source: 0.5692) | 0.5212 (best open-source: 0.4010) |
| Non-object-only | 109 | 0.5446 (best open-source: 0.5530) | 0.3853 (best open-source: 0.2273) |
| Mixed | 1,263 | 0.5791 (best open-source: 0.4896) | 0.4575 (best open-source: 0.3284) |

These results show that non-object targets can be evaluated independently within the existing benchmark. GPT-6.1-Sol achieves higher 2D localization scores than the available open-source baselines, while its anchor-selection score on non-object-only interactions remains close to Qwen3-VL-30B, indicating residual difficulty in semantic target selection. The evaluation uses representative points for non-object anchors and contains no manually annotated region boundaries or area supervision; thus, it measures representative-point selection and localization. The 2D results use GPT's declared-coordinate-space remapping and should be interpreted descriptively across models.
