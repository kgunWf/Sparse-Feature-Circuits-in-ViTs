# Feature Catalog - Layer 9 (CLIP ViT-B/32)

Status: AI first pass complete; 10 features independently reviewed. See `manual_annotation_review.md`.

Selection: top 50 finite features by top-5 Monosemanticity Score. Categories use the handoff decision rules after reviewing the top-20 patch crops and top-three CLIP labels.

| Feature idx | MS Score | CLIP label (top-3) | Category |
|---:|---:|---|---|
| 835 | 0.990 | smooth surface / blotchy texture / coarse texture | color |
| 48077 | 0.983 | spinet / keyboard / grand piano | part |
| 36083 | 0.980 | featheredge / lacelike texture / fabric texture | texture |
| 29569 | 0.975 | spinet / keyboard / grand piano | part |
| 37953 | 0.975 | grass / grass fern / camouflage pattern | scene |
| 6098 | 0.975 | spinet / keyboard / panpipe | part |
| 33231 | 0.974 | leopard / camouflage pattern / leopardess | semantic |
| 30794 | 0.974 | spinet / coarse texture / panpipe | part |
| 5787 | 0.973 | spinet / panpipe / keyboard | part |
| 4802 | 0.973 | tailorbird / grassfinch / puffbird | unclear |
| 8632 | 0.973 | tailorbird / grassfinch / puffbird | unclear |
| 26274 | 0.973 | tailorbird / grassfinch / puffbird | unclear |
| 40310 | 0.973 | tailorbird / grassfinch / puffbird | unclear |
| 47904 | 0.973 | tailorbird / grassfinch / puffbird | unclear |
| 39011 | 0.973 | tailorbird / grassfinch / puffbird | unclear |
| 39066 | 0.972 | oboe / panpipe / banded texture | part |
| 21069 | 0.972 | spinet / panpipe / keyboard | part |
| 48641 | 0.972 | spinet / marimba / keyboard | part |
| 33723 | 0.972 | spinet / keyboard / panpipe | part |
| 37241 | 0.972 | spinet / dovetail / lacelike texture | part |
| 8371 | 0.972 | limb / tuft / edge | unclear |
| 9175 | 0.972 | spinet / panpipe / lacelike texture | part |
| 21093 | 0.972 | spinet / keyboard / coarse texture | part |
| 23189 | 0.972 | wrist / spinet / banded texture | part |
| 3234 | 0.971 | spinet / keyboard / panpipe | part |
| 12160 | 0.971 | spinet / panpipe / keyboard | part |
| 27906 | 0.971 | spinet / keyboard / panpipe | part |
| 33056 | 0.971 | spinet / keyboard / grand piano | part |
| 39201 | 0.971 | spinet / keyboard / marimba | part |
| 24086 | 0.971 | spinet / keyboard / marimba | part |
| 33171 | 0.970 | snout / bark / dogie | unclear |
| 25079 | 0.970 | spinet / acoustic guitar / cittern | part |
| 38282 | 0.970 | spinet / panpipe / keyboard | part |
| 6767 | 0.970 | spinet / keyboard / panpipe | part |
| 10324 | 0.970 | spinet / keyboard / marimba | part |
| 5183 | 0.970 | keyboard / spinet / marimba | part |
| 32139 | 0.970 | lacelike texture / coarse texture / bark | part |
| 41822 | 0.970 | spinet / keyboard / marimba | part |
| 1795 | 0.969 | spinet / keyboard / panpipe | part |
| 10730 | 0.969 | spinet / keyboard / panpipe | part |
| 37912 | 0.969 | tailorbird / grassfinch / puffbird | unclear |
| 29133 | 0.969 | tailorbird / grass fern / grassfinch | unclear |
| 3958 | 0.969 | spinet / keyboard / marimba | part |
| 39204 | 0.969 | spinet / panpipe / oboe | part |
| 45211 | 0.969 | spinet / rectangular shape / lacelike texture | part |
| 9565 | 0.969 | spinet / panpipe / grooved texture | part |
| 41375 | 0.969 | spinet / panpipe / keyboard | part |
| 32238 | 0.969 | spinet / marimba / panpipe | part |
| 48273 | 0.968 | lacelike texture / wrist / loupe | part |
| 27759 | 0.968 | insect / wing / homopterous insect | unclear |

Category counts: texture=1, color=1, part=35, scene=1, semantic=1, unclear=11.
