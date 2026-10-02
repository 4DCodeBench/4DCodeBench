# VLM-as-judge

This directory implements the Vision-Language Model (VLM) evaluation pipelines: Visual Question Answering (VQA) and pairwise preference evaluation (Elo).

`judge.py vqa` queries a vision-language model with yes/no questions about a system's rendered video. `judge.py pairwise` presents the reference video alongside two candidate renders and evaluates which is closer. Both use `google/gemini-3.8-flash` at temperature 0.

    export OPENROUTER_API_KEY=...

    python3 vlm_judge/judge.py vqa      --system <system> --out answers.jsonl
    python3 vlm_judge/judge.py score    answers.jsonl

    python3 vlm_judge/judge.py pairwise --out pairs.jsonl
    python3 vlm_judge/judge.py elo      pairs.jsonl --boot 200

Run from the repository root; both read videos from `cases/` and `runs/`, see
[Where the videos come from](#where-the-videos-come-from).

## Requirements

Requires Python 3.9+ with `ffmpeg` and `ffprobe` on `PATH`. The scripts use only the Python standard library. Both `ffmpeg` (for frame decoding) and `ffprobe` (for frame counts) are required; `judge.py` reports an error if either is missing.

The repository's conda environment provides all dependencies:

    conda env create -f environment.yml && conda activate 4dcodebench

Alternatively, any Python 3.9+ environment with system-installed ffmpeg (e.g. `apt install ffmpeg`) can be used.

## Frames

The two runs sample differently, because they ask different things.

| | frames | width | why |
|---|---|---|---|
| `vqa` | 16 per video | 768 px | first frame, final frame, 14 spread evenly between |
| `pairwise` | 3 fps | 512 px | three videos in one call, so the rate sets the cost |

The VQA endpoints are fixed rather than sampled. The question set asks about the first and
the final frame by name, and a uniform sample over a 40-frame clip lands on neither.

## Evaluation Calls

`vqa` issues one API call per scene, batching all questions for that scene with the sampled video frames to avoid redundant image tokens. A scene averages about 18k prompt tokens, about $0.02, so one system over 200 scenes costs about $4.

The prompt instructs the model that the reconstruction may be incomplete, and that an object entirely absent from the video must be answered as `Missing` rather than `No`. This prevents unrendered objects from receiving credit on negative questions.

`pairwise` issues one API call per pair, using a uniform prompt across all scenes. Presentation order (Video A vs. Video B) is balanced to prevent order-dependent position bias. A comparison averages about 69k prompt tokens, about $0.055; the default `--pairs 3` buys up to 600 comparisons over 200 scenes, about $33.

Comparisons are sampled via round-robin matching across systems. `--pairs` (default: 3) selects the least-compared pairs in each round to balance comparisons across systems; `--pairs 0` evaluates all pairs in the round.

API calls are skipped for deterministic outcomes: if a system produces no render or fails the Layer 1 gate (see [Validation Gates](#validation-gates)), it loses the matchup; if both fail, the matchup is scored as a tie.

`--max-tokens` defaults to 1024 to accommodate model reasoning traces before the final verdict. Pass `--max-tokens 256` to match the exact setting used in the paper.

## The prompts

Both are fixed. Neither varies by scene, by category, or by system.

`vqa`, one call per scene, with that scene's questions numbered into it:

```
These are 16 frames from one video, in order. The first is the video's first frame
and the last is its final frame; the rest are sampled evenly from in between.

Answer each of the following N questions about that video. Each is a yes/no question
about what is physically visible.

These frames come from a 3D reconstruction that may be incomplete, so an object a
question names may be absent from the video entirely.
If an object a question asks about is NOT IN THE VIDEO AT ALL, answer 'Missing' for
that question.
Missing is only about objects that are not there. If the objects ARE there but the
thing described does not happen, or looks different from the description, that is an
ordinary No, not Missing.
Only answer Yes or No when you can see the things the question is about.

1. <question>
2. <question>
...

Reply with one line per question, in order, formatted exactly as:
1. Yes
2. No
3. Missing
(and so on). Give no other text.
```

The `Missing` instruction is placed at the top of the prompt to govern every question in that scene, including the presence question, preventing unrendered objects from receiving default negative credit.

`pairwise`, one call per comparison:

```
You are shown a reference recording of a physical event, then two attempts by coding
models to reproduce it: video A and video B.

Which attempt is closer to the reference? Consider the geometry of the objects in the
scene, how they move, the physics, the collisions, the deformations, where things end up.

Answer with one word: A or B.
```

## Validation Gates

Two validation gates evaluate whole-scene validity:

- **Layer 1 Filter**: Runs before model invocation. A render fails Layer 1 if it contains insufficient visual structure (edge density < 1.0%) or is dominated by a single uniform color (dominant color share > 0.95), evaluated every eighth frame. Gated scenes are not queried via the API and are marked as incorrect in scoring. Use `--no-layer1` to bypass this filter.
- **Presence Gate**: The second gate consists of an initial presence question per scene (`Q0`, category `presence`). It verifies whether the scene's primary dynamic object was reconstructed, even crudely (e.g. *"In the first frame, is there something that looks like a wine glass? Answer Yes even if it is crude, roughly shaped, or a different colour."*). If a render fails this check, subsequent questions cannot reliably evaluate physical motion and are scored as incorrect. The presence question is excluded from the accuracy denominator. Use `--no-presence` to score answered questions regardless.

## Scoring & Elo Ratings

`score` computes accuracy over all released benchmark questions, counting unrendered or gated scenes as incorrect. Because the dataset has an affirmative prior (1,079 of 1,251 questions have ground-truth 'Yes', yielding an all-Yes baseline of 0.863), evaluation should primarily reference the balanced accuracy column.

`elo` fits a Bradley-Terry model via iterative scaling and reports ratings as $400 \log_{10} p$ centered around zero (geometric mean normalized to 1). A tie counts as 0.5 wins for each side. Crashed API calls are dropped rather than scored as ties.

Because ratings are centered relative to the evaluated population, **adding or removing systems shifts absolute ratings**; only relative score differences between systems are directly comparable across different fits.

`--boot` (default: 200 iterations, `0` to skip) computes 95% bootstrap confidence intervals by resampling **tasks rather than comparisons**, preserving correlation across comparisons that share identical task conditions and renders.

## vqa_questions_and_ontology_final.json

One file: the ontology and every question, per scene. 200 scenes, 1251 scored questions
and 200 presence questions.

    {"generated", "split", "n_scenes", "n_questions",
     "notes":      what each field means and how the run was scored,
     "vocabulary": the 4 families and 10 materials, each with its definition,
     "counts":     scenes per family, per material, questions per category,
     "scenes":     [{id, kind, description, tags, families, materials,
                     material_families, bodies, n_materials, dynamics_regime,
                     flags, questions: [{id, question, options, answer, category}]}]}

The ontology includes two distinct categorization levels by design: `families` reflects coarse annotator tags (used in dataset summary tables), while `material_families` reflects resolved constitutive materials. They differ on nine scenes (e.g., scenes tagged `plastic` whose material resolves to `viscoplastic`). Both classifications are multi-label.

Question categories: `first_frame` (371), `last_frame` (354), `key_event` (304),
`contact_response` (218), `material` (4), and `presence` (200), one per scene, the gate.
`n_questions` is 1251 and excludes the presence questions, as does the accuracy
denominator.

## Where the videos come from

Both commands read the repository's own trees:

    cases/<kind>/<case>/reference.mp4                              the reference
    runs/<kind>/<case>/<system>/1/workspace/world/render.mp4       a system's render

`<system>` is the run directory's name, `<agent>-<model>-<effort>`. `python
scripts/download_data.py --videos-only` fills `cases/`, and inference fills `runs/`.
`--cases` and `--runs` point elsewhere.

`vqa` judges one system at a time, `--system`. `pairwise` is comparative -- a rating
places a system against the others -- so it takes every system under `runs/`, or the
ones named by `--systems`. A run that delivered no `render.mp4` counts as missing, and
the pairwise rule scores the absence as a loss.

To rate a new system against the ones in the paper, download their renders and place
them under `runs/`:

    hf download 4DCodeBench/Results --repo-type dataset --local-dir results
    python scripts/place_renders.py results

`results/pairwise_paper.jsonl` holds the paper's 2221 comparisons among those systems.
Start `--out` from a copy of it: `pairwise` skips any comparison already there, and
`elo` fits the paper's comparisons together with yours. The pairs `pairwise` picks per
scene can include pairs of the paper's systems, so the cost above still applies.

    cp vlm_judge/results/pairwise_paper.jsonl pairs.jsonl
    python3 vlm_judge/judge.py pairwise --out pairs.jsonl
    python3 vlm_judge/judge.py elo      pairs.jsonl

Both commands append results to `--out` and skip already-evaluated entries, allowing interrupted runs to resume seamlessly.
