# Manual Annotation Review

Completed 2026-06-12 from the six annotated PNGs in `report/figures/manual_review/`.
The first label was used for slash-separated responses. `text / color` was
normalized to `color`, the first response that belongs to the six-category scheme.

| Model | Reviewed | Matches with AI | Agreement |
|---|---:|---:|---:|
| DINO ViT-B/16 | 30 | 13 | 43.3% |
| CLIP ViT-B/32 | 30 | 4 | 13.3% |

## Transcription

| Model | Layer | Human labels, in catalog order | Matches |
|---|---:|---|---:|
| DINO | 4 | `21380=color; 14985=semantic; 35131=color; 32021=unclear; 41084=color; 26510=unclear; 48979=unclear; 10634=unclear; 22178=color; 27473=color` | 4/10 |
| DINO | 6 | `11782=color; 37726=color; 44648=color; 41715=color; 40170=color; 13412=part; 16287=part; 15228=color; 11107=color; 38876=texture` | 4/10 |
| DINO | 9 | `54=color; 35641=texture; 29161=texture; 5778=texture; 2163=unclear; 23730=semantic; 46136=part; 27931=texture; 30084=color; 4249=semantic` | 5/10 |
| CLIP | 4 | `38372=color; 8255=color; 42102=color; 27542=color; 19415=texture; 31196=color; 39705=color; 25795=part; 31788=part; 6743=color` | 2/10 |
| CLIP | 6 | `21681=color; 40875=color; 24186=texture; 31971=color; 47411=color; 19072=color; 31449=unclear; 17521=color; 24070=color; 31697=color` | 0/10 |
| CLIP | 9 | `835=unclear; 48077=color; 36083=texture; 29569=color; 37953=texture; 6098=semantic; 33231=semantic; 30794=semantic; 5787=semantic; 4802=color` | 2/10 |

Agreement is exact category agreement against the AI first-pass `Category` column.
Only these 30 features per model were independently double-coded.
