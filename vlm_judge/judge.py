#!/usr/bin/env python3
"""The two VLM-as-judge runs behind the 4DCodeBench paper, in one script.

    judge.py vqa      --system <agent>-<model>-<effort> --out answers.jsonl
    judge.py pairwise --out pairs.jsonl
    judge.py score    answers.jsonl
    judge.py elo      pairs.jsonl

`vqa` asks the questions in vqa_questions_and_ontology_final.json about one system's
renders: one call per scene, 16 frames, every question about that scene in the same call.
`pairwise` shows the judge a reference and two systems' renders and asks which is closer:
one call per pair, one prompt for every comparison, 3 fps.

Both need OPENROUTER_API_KEY, and ffmpeg and ffprobe on PATH. No Python
packages: stdlib only.
"""
import argparse, base64, collections, json, math, os, random, re, shutil
import subprocess, sys, tempfile, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "vqa_questions_and_ontology_final.json"
API = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3.8-flash"
LOCK = threading.Lock()

# ----------------------------------------------------------------- the prompts

# VQA. One call per scene, so the rule about absent objects has to be stated once for
# the whole batch. Missing is not a third truth value: it separates "the object is not
# in this reconstruction" from "the object is there and does not do this", which a
# broken render would otherwise collect as a lucky No.
MISSING_RULE = (
    "\nThese frames come from a 3D reconstruction that may be incomplete, so an object a "
    "question names may be absent from the video entirely.\n"
    "If an object a question asks about is NOT IN THE VIDEO AT ALL, answer 'Missing' for "
    "that question.\n"
    "Missing is only about objects that are not there. If the objects ARE there but the "
    "thing described does not happen, or looks different from the description, that is an "
    "ordinary No, not Missing.\n"
    "Only answer Yes or No when you can see the things the question is about.\n")

# Pairwise. The same prompt for every comparison, on every scene: no per-scene rubric and
# no per-axis variant, so one rating scale comes out the other end.
PAIRWISE = (
    "You are shown a reference recording of a physical event, then two attempts by coding "
    "models to reproduce it: video A and video B.\n\n"
    "Which attempt is closer to the reference? Consider the geometry of the objects in the "
    "scene, how they move, the physics, the collisions, the deformations, where things end "
    "up.\n\n"
    "Answer with one word: A or B.")


def vqa_prompt(questions, n_frames):
    head = (f"These are {n_frames} frames from one video, in order. The first is the video's "
            "first frame and the last is its final frame; the rest are sampled evenly from "
            "in between.\n\n"
            f"Answer each of the following {len(questions)} questions about that video. "
            "Each is a yes/no question about what is physically visible.\n\n")
    body = "\n".join(f"{i}. {q['question']}" for i, q in enumerate(questions, 1))
    return (head + MISSING_RULE + "\n" + body +
            "\n\nReply with one line per question, in order, formatted exactly as:\n"
            "1. Yes\n2. No\n3. Missing\n(and so on). Give no other text.")

# ------------------------------------------------------------------- the frames


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def n_frames(video):
    for args in (["-count_frames", "-show_entries", "stream=nb_read_frames"],
                 ["-show_entries", "stream=nb_frames"]):
        out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0"] + args +
                   ["-of", "csv=p=0", str(video)]).strip().rstrip(",")
        if out.isdigit() and int(out) > 0:
            return int(out)
    raise RuntimeError(f"{video}: cannot count frames")


def even_frames(video, n=16, width=768, cache=None):
    """The first frame, the last frame, and n-2 spread evenly through the middle.

    The endpoints are not negotiable: the question set asks about the first and the final
    frame by name, and a uniform sample over a short clip misses both.
    """
    out = Path(cache) if cache else Path(tempfile.mkdtemp())
    out.mkdir(parents=True, exist_ok=True)
    have = sorted(out.glob("f*.jpg"))
    if len(have) == n:
        return have
    for f in have:
        f.unlink()
    total = n_frames(video)
    if total < n:
        idx = list(range(total))
    else:
        mid = sorted({round(1 + i * (total - 3) / (n - 3)) for i in range(n - 2)}) if n > 2 else []
        idx = [0] + mid + [total - 1]
    sel = "+".join(rf"eq(n\,{i})" for i in idx)
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(video),
          "-vf", f"select='{sel}',scale={width}:-2", "-vsync", "0", "-q:v", "3",
          str(out / "f%03d.jpg")])
    return sorted(out.glob("f*.jpg"))


def fps_frames(video, fps=3.0, width=512):
    """Every 1/fps of a second. Three of these go in one pairwise call, so the rate is
    what keeps the call affordable; 3 fps is what the study ran at."""
    out = Path(tempfile.mkdtemp())
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(video),
          "-vf", f"fps={fps},scale={width}:-2", "-q:v", "4", str(out / "f%03d.jpg")])
    return sorted(out.glob("*.jpg"))


EDGE_MIN, FLAT_MAX = 1.0, 0.95


def layer1(video, every=8):
    """Edge density and single-colour share, the cheap local gate.

    A render fails when it has almost no edges (nothing in it) or is almost entirely one
    colour (blank or flat-lit). Both are read from every `every`-th frame, which is enough
    and keeps this quick. Thresholds are the ones the released run used, chosen so that no
    render from the strongest systems is caught.

    The two statistics come out within about two points of edge density of the released
    run's, the difference being which bicubic resize the frames went through. The closest
    render in that run still sits 3 points clear of the edge threshold, so no verdict in it
    changes; a render near a threshold could read either way.
    """
    def frames(w, h):
        """Every `every`-th frame, scaled to w x h, as raw RGB."""
        return subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video),
             "-vf", rf"select='not(mod(n\,{every}))',scale={w}:{h}",
             "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, check=True).stdout

    try:
        coarse = frames(160, 90)
        fine = frames(96, 54)
    except Exception:
        return {"edge": None, "top_colour": None, "gated": False}
    if not coarse or not fine:
        return {"edge": None, "top_colour": None, "gated": False}

    # Edges: how often two neighbouring pixels differ by more than 6 levels of grey, once
    # across and once down, averaged. Grey is BT.601 luma computed here rather than asked
    # of ffmpeg, which would pick a matrix from the input's tagged colourspace and make the
    # number depend on how the render was muxed.
    W, H = 160, 90
    hit, seen = [0, 0], [0, 0]
    for f in range(len(coarse) // (W * H * 3)):
        px = coarse[f * W * H * 3:(f + 1) * W * H * 3]
        grey = bytes((px[i] * 299 + px[i + 1] * 587 + px[i + 2] * 114) // 1000
                     for i in range(0, W * H * 3, 3))
        for y in range(H):
            row = grey[y * W:(y + 1) * W]
            hit[0] += sum(1 for a, b in zip(row, row[1:]) if abs(a - b) > 6)
            seen[0] += W - 1
        for y in range(H - 1):
            up, down = grey[y * W:(y + 1) * W], grey[(y + 1) * W:(y + 2) * W]
            hit[1] += sum(1 for a, b in zip(up, down) if abs(a - b) > 6)
            seen[1] += W
    edge = (hit[0] / seen[0] + hit[1] / seen[1]) / 2 * 100

    # Flatness: the share the commonest colour takes, colour quantised to 8 levels per
    # channel so that gradients and dither do not read as variety.
    CW, CH = 96, 54
    npx = CW * CH
    shares = []
    for f in range(len(fine) // (npx * 3)):
        px = fine[f * npx * 3:(f + 1) * npx * 3]
        n = collections.Counter((px[i] // 32) * 64 + (px[i + 1] // 32) * 8 + px[i + 2] // 32
                                for i in range(0, npx * 3, 3))
        shares.append(max(n.values()) / npx)
    top = sum(shares) / len(shares)

    return {"edge": round(edge, 3), "top_colour": round(top, 3),
            "gated": edge < EDGE_MIN or top > FLAT_MAX}


def img(p):
    return {"type": "image_url", "image_url":
            {"url": "data:image/jpeg;base64," + base64.b64encode(Path(p).read_bytes()).decode()}}

# ---------------------------------------------------------------------- the call


def call(model, content, max_tokens=4096, retries=8):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens,
                       "usage": {"include": True},
                       "messages": [{"role": "user", "content": content}]}).encode()
    hdr = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
           "Content-Type": "application/json"}
    err = "no attempt"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API, data=body, headers=hdr)
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            if "error" in d:
                raise RuntimeError(str(d["error"])[:200])
            return d["choices"][0]["message"].get("content") or "", d.get("usage") or {}
        except urllib.error.HTTPError as e:
            if e.code in (401, 402, 403):           # auth or credit: retrying cannot help
                sys.exit(f"fatal {e.code}: {e.read().decode()[:200]}")
            err = f"HTTP {e.code}"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        time.sleep(min(90, 3 * 2 ** attempt))
    return None, {"error": err}

# ----------------------------------------------------------------------- the VQA


def parse_vqa(text, k):
    out = [None] * k
    if not text:
        return out
    for m in re.finditer(r"^\s*(\d+)\s*[.):]\s*(yes|no|missing)\b", text, re.I | re.M):
        i = int(m.group(1)) - 1
        if 0 <= i < k:
            out[i] = m.group(2).capitalize()
    if all(v is None for v in out):     # the model ignored the numbering; fall back to order
        for i, v in enumerate(re.findall(r"\b(yes|no|missing)\b", text, re.I)[:k]):
            out[i] = v.capitalize()
    return out


def load_scenes(path, split=None, only=None):
    d = json.load(open(path))
    s = d["scenes"]
    if split:
        s = [x for x in s if x["kind"] == split]
    if only:
        s = [x for x in s if x["id"] in only]
    return s


# The repository's own trees: cases/<kind>/<case>/reference.mp4, and each run's render
# under runs/<kind>/<case>/<system>/, where <system> is the run directory's name,
# <agent>-<model>-<effort>. The paper's runs are trial 1.
KINDS = ("real", "synthetic")
RENDER = "1/workspace/world/render.mp4"


def find_reference(cases, scene_id):
    for kind in KINDS:
        p = Path(cases) / kind / scene_id / "reference.mp4"
        if p.exists():
            return p
    return None


def find_render(runs, system, scene_id):
    for kind in KINDS:
        p = Path(runs) / kind / scene_id / system / RENDER
        if p.exists():
            return p
    return None


def list_systems(runs):
    """Every system with at least one render under runs/."""
    return sorted({p.parts[-5] for p in Path(runs).glob(f"*/*/*/{RENDER}")})


def cmd_vqa(a):
    scenes = load_scenes(a.questions, a.split, set(a.scenes.split(",")) if a.scenes else None)
    runs = Path(a.runs)
    done = set()
    if Path(a.out).exists():
        done = {(r["scene"], r["id"]) for r in map(json.loads, open(a.out)) if r.get("pred")}

    jobs, missing, gated = [], [], []
    for s in scenes:
        qs = [q for q in s["questions"] if (s["id"], q["id"]) not in done]
        if not qs:
            continue
        v = find_render(runs, a.system, s["id"])
        if v is None:
            missing.append(s["id"])
            continue
        # Layer 1 first: a render with nothing in it scores wrong either way, and
        # gating it here is free where a judge call is not.
        if not a.no_layer1 and layer1(v)["gated"]:
            gated.append(s["id"])
            continue
        jobs.append((s, qs, v))

    print(f"{len(jobs)} scenes, {sum(len(q) for _, q, _ in jobs)} questions, "
          f"one call per scene, {a.frames} frames, judge={a.model}")
    if missing:
        # A scene with no video is not an absence of evidence: score it wrong. The
        # released numbers count it that way, so the run records it rather than dropping it.
        print(f"  {len(missing)} scenes have no render: {' '.join(missing[:6])}"
              f"{' ...' if len(missing) > 6 else ''}")
    if gated:
        print(f"  {len(gated)} scenes fail layer 1 and are not judged: "
              f"{' '.join(gated[:6])}{' ...' if len(gated) > 6 else ''}")
    if not jobs:
        return

    cache = Path(a.out).resolve().parent / "frame_cache"

    def work(job):
        s, qs, v = job
        try:
            fr = even_frames(v, a.frames, a.width, cache / s["id"])
        except Exception as e:
            return s, qs, [None] * len(qs), f"FRAMES {type(e).__name__}: {e}", {}
        content = [img(p) for p in fr]
        content.append({"type": "text", "text": vqa_prompt(qs, len(fr))})
        text, usage = call(a.model, content, a.max_tokens)
        return s, qs, parse_vqa(text, len(qs)), text, {**usage, "n_frames": len(fr)}

    t0, n = time.time(), 0
    with open(a.out, "a") as out, ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            s, qs, preds, text, usage = f.result()
            nq = max(1, len(qs))
            share = {k: v / nq for k, v in usage.items() if isinstance(v, (int, float))}
            with LOCK:
                for q, pred in zip(qs, preds):
                    out.write(json.dumps({
                        "scene": s["id"], "kind": s["kind"], "id": q["id"],
                        "category": q["category"], "answer": q["answer"], "pred": pred,
                        "correct": pred == q["answer"], "missing": pred == "Missing",
                        "system": a.system, "judge": a.model, "per_call": len(qs),
                        "usage": share, "raw": (text or "")[:200]}) + "\n")
                out.flush()
                n += len(qs)
                print(f"  {n}/{sum(len(q) for _, q, _ in jobs)}  {time.time()-t0:.0f}s",
                      end="\r", flush=True)
    print(f"\n{a.out}  {time.time()-t0:.0f}s")

# ------------------------------------------------------------------ the pairwise


def matching(systems, rnd):
    """Round `rnd` of a round-robin: every system in one pair, one sitting out if odd."""
    m = sorted(systems)
    if len(m) % 2:
        bye = m[rnd % len(m)]
        m = [x for x in m if x != bye]
    fixed, rest = m[0], m[1:]
    k = rnd % len(rest)
    r = [fixed] + rest[k:] + rest[:k]
    return [(r[i], r[len(r) - 1 - i]) for i in range(len(r) // 2)]


def cmd_pairwise(a):
    scenes = load_scenes(a.questions, a.split, set(a.scenes.split(",")) if a.scenes else None)
    cases, runs = Path(a.cases), Path(a.runs)
    systems = sorted(a.systems.split(",")) if a.systems else list_systems(runs)
    if len(systems) < 2:
        sys.exit(f"need two or more systems under {runs}, found {systems}")

    done = set()
    if Path(a.out).exists():
        done = {(r["scene"], r["a"], r["b"]) for r in map(json.loads, open(a.out))
                if r.get("winner") or r.get("how") == "draw"}

    # Balance the A slot across systems. The judge leans toward whichever video it sees
    # first, so a design that always puts the same system on the left measures that lean
    # as if it were skill.
    rng = random.Random(a.seed)
    slot_a = collections.Counter()
    met = collections.Counter()          # comparisons bought per system so far
    jobs = []
    for i, s in enumerate(scenes):
        ref = find_reference(cases, s["id"])
        if ref is None:
            continue
        have = {m: find_render(runs, m, s["id"]) for m in systems}
        # Of this scene's round of the round-robin, buy the pairs whose two systems have
        # been compared least so far, so the budget levels the design across systems.
        pairs = sorted(matching(systems, i), key=lambda xy: (met[xy[0]] + met[xy[1]], rng.random()))
        for x, y in pairs[:a.pairs] if a.pairs else pairs:
            met[x] += 1; met[y] += 1
            if have[x] is None and have[y] is None:
                continue                        # neither delivered: nothing to compare
            left, right = (x, y) if slot_a[x] <= slot_a[y] else (y, x)
            slot_a[left] += 1
            if (s["id"], left, right) in done or (s["id"], right, left) in done:
                continue
            jobs.append((s, ref, left, have[left], right, have[right]))

    print(f"{len(jobs)} comparisons over {len(systems)} systems, {a.fps} fps, judge={a.model}")
    if not jobs:
        return

    gate = {}

    def gated(p):
        if p not in gate:
            gate[p] = layer1(p)["gated"]
        return gate[p]

    def work(job):
        s, ref, la, pa, lb, pb = job
        # Decided without a call: a missing render loses, a render layer 1 gates loses,
        # and two gated renders draw, since they failed the same way.
        if pa is None or pb is None:
            return s, la, lb, "B" if pa is None else "A", "", {}, (0, 0, 0), "coverage"
        ga, gb = gated(pa), gated(pb)
        if ga or gb:
            pick = None if ga and gb else "B" if ga else "A"
            return s, la, lb, pick, "", {}, (0, 0, 0), "draw" if ga and gb else "gated"
        try:
            fr, fa, fb = (fps_frames(p, a.fps, a.width) for p in (ref, pa, pb))
        except Exception as e:
            return s, la, lb, None, f"FRAMES {type(e).__name__}", {}, (0, 0, 0), "error"
        content = [{"type": "text", "text": f"The REFERENCE, {len(fr)} frames in order:"}]
        content += [img(p) for p in fr]
        content += [{"type": "text", "text": f"VIDEO A, {len(fa)} frames in order:"}]
        content += [img(p) for p in fa]
        content += [{"type": "text", "text": f"VIDEO B, {len(fb)} frames in order:"}]
        content += [img(p) for p in fb]
        content += [{"type": "text", "text": PAIRWISE}]
        text, usage = call(a.model, content, a.max_tokens)
        pick = None
        for w in re.findall(r"\b([AB])\b", (text or "").upper()):
            pick = w
            break
        return s, la, lb, pick, text, usage, (len(fr), len(fa), len(fb)), \
            "judged" if pick else "error"

    t0, n = time.time(), 0
    with open(a.out, "a") as out, ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            s, la, lb, pick, text, usage, nf, how = f.result()
            with LOCK:
                out.write(json.dumps({
                    "scene": s["id"], "kind": s["kind"], "a": la, "b": lb,
                    "winner": {"A": la, "B": lb}.get(pick),
                    "how": how, "judge": a.model,
                    "fps": a.fps, "frames": {"ref": nf[0], "a": nf[1], "b": nf[2]},
                    "usage": usage, "raw": (text or "")[:120]}) + "\n")
                out.flush()
                n += 1
                print(f"  {n}/{len(jobs)}  {time.time()-t0:.0f}s", end="\r", flush=True)
    print(f"\n{a.out}  {time.time()-t0:.0f}s")

# ------------------------------------------------------------------- the scoring


def cmd_score(a):
    """Accuracy over every released question, not over the ones a system was asked.

    A scene a system never rendered, or a call the judge could not answer, counts wrong.
    That is what makes systems that failed on different scenes comparable: the alternative
    scores each one on its own easier subset.

    Two gates act on whole scenes. A scene whose render fails layer 1 is never judged, so
    its questions are simply absent from the answers and count wrong. A scene whose
    presence question is answered wrong has its remaining questions counted wrong even
    though the judge answered them: the main object is not in the render, so those answers
    are not evidence about physics. The presence question is a gate and is never itself in
    the denominator.
    """
    rows = [json.loads(l) for l in open(a.answers)]
    # Gold is keyed by question, not counted, so a file holding answers to questions
    # outside the released set -- an earlier run of a wider draft, or several systems
    # concatenated -- is scored on the released set rather than inflated by the extras.
    gold = {(s["id"], q["id"]): q["answer"]
            for s in json.load(open(a.questions))["scenes"]
            for q in s["questions"] if q["category"] != "presence"}
    n_all = len(gold)
    n_yes = sum(1 for v in gold.values() if v == "Yes")
    n_no = n_all - n_yes

    by = collections.defaultdict(list)
    for r in rows:
        by[r["system"]].append(r)

    print(f"{n_all} questions, {n_yes} gold Yes, {n_no} gold No; "
          f"always-Yes baseline {n_yes / n_all:.3f}")
    print(f"{'system':<24} {'acc':>7} {'yes':>7} {'no':>7} {'bal':>7} {'miss':>6} {'asked':>6}")

    out = []
    for m, rs in by.items():
        # a scene whose presence question is wrong has its other answers set aside: the
        # object is not in the render, so they are not evidence about physics
        dead = set() if a.no_presence else {
            r["scene"] for r in rs if r["category"] == "presence" and not r["correct"]}
        ans = {}
        for r in rs:
            k = (r["scene"], r["id"])
            if k not in gold:
                continue
            ans[k] = (bool(r["correct"]) and r["scene"] not in dead, bool(r.get("missing")))
        hit = [k for k, (ok, _) in ans.items() if ok]
        ay = sum(1 for k in hit if gold[k] == "Yes") / max(1, n_yes)
        an = sum(1 for k in hit if gold[k] == "No") / max(1, n_no)
        out.append((len(hit), m, len(hit) / n_all, ay, an, (ay + an) / 2,
                    sum(1 for _, (_, mi) in ans.items() if mi), len(ans)))
    for _, m, acc, ay, an, bal, miss, asked in sorted(out, reverse=True):
        print(f"{m:<24} {acc:7.3f} {ay:7.3f} {an:7.3f} {bal:7.3f} {miss:6d} {asked:6d}")


def _bt(rows, prior=1.0, iters=9000):
    """Bradley-Terry by iterative scaling. A draw is half a win each.

    Only a decided comparison counts. `draw` is a real outcome -- both sides failed the
    same way -- but a call that crashed is not a draw, and counting one as such would
    hand half a win to each side for a missing measurement.
    """
    models = sorted({m for r in rows for m in (r["a"], r["b"])})
    w, g = collections.Counter(), collections.defaultdict(collections.Counter)
    for r in rows:
        if r["how"] == "draw":
            w[r["a"]] += .5; w[r["b"]] += .5
        elif r["how"] in ("judged", "coverage", "gated") and r.get("winner"):
            w[r["winner"]] += 1
        else:
            continue
        g[r["a"]][r["b"]] += 1; g[r["b"]][r["a"]] += 1
    p = {m: 1.0 for m in models}
    for _ in range(iters):
        new = {}
        for m in models:
            den = sum(x / (p[m] + p[o]) for o, x in g[m].items()) + prior / (p[m] + 1.0)
            new[m] = (w[m] + prior / 2) / den
        s_ = math.prod(new.values()) ** (1 / len(new))
        p = {m: v / s_ for m, v in new.items()}
    return {m: 400 * math.log10(p[m]) for m in models}, w, g


def _pct(v, q):
    v = sorted(v)
    i = (len(v) - 1) * q / 100
    lo = int(i)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def cmd_elo(a):
    """Bradley-Terry by iterative scaling, then 400 log10 p. A draw is half a win.

    The scale is centred on zero by normalising p to geometric mean 1, so a rating is a
    distance from the field's average and not from an arbitrary anchor. That also means
    the scale depends on who is in the field: adding a system moves every rating, and
    only the differences are comparable across two fits.

    The interval resamples TASKS, not comparisons. Several comparisons on one task share
    its difficulty and its render, so they are not independent evidence, and bootstrapping
    comparisons would report an interval several times too narrow.
    """
    rows = [json.loads(l) for l in open(a.pairs)]
    elo, w, g = _bt(rows, a.prior)

    ci = {}
    if a.boot:
        by = collections.defaultdict(list)
        for r in rows:
            by[r.get("scene") or r.get("case")].append(r)
        tasks = sorted(by)
        rng = random.Random(a.seed)
        draws = collections.defaultdict(list)
        for _ in range(a.boot):
            pick = [r for t in (rng.choice(tasks) for _ in tasks) for r in by[t]]
            for m, v in _bt(pick, a.prior)[0].items():
                draws[m].append(v)
        ci = {m: (_pct(v, 2.5), _pct(v, 97.5)) for m, v in draws.items()}

    print(f"{len(rows)} comparisons, {len(elo)} systems"
          + (f", {a.boot}-draw interval over tasks" if a.boot else ""))
    head = f"{'system':<24} {'elo':>8}" + ("      95% interval" if ci else "") \
           + f" {'wins':>7} {'games':>6}"
    print(head)
    for m in sorted(elo, key=lambda x: -elo[x]):
        band = f"  [{ci[m][0]:7.0f},{ci[m][1]:7.0f}]" if ci else ""
        print(f"{m:<24} {elo[m]:8.1f}{band} {w[m]:7.1f} {sum(g[m].values()):6d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--questions", default=str(DATA))
        p.add_argument("--split", choices=("real", "synthetic"))
        p.add_argument("--scenes", default="", help="comma-separated scene ids")
        p.add_argument("--model", default=MODEL, help="the judge")
        p.add_argument("--workers", type=int, default=3)

    v = sub.add_parser("vqa", help="answer the question set about one system's renders")
    common(v)
    v.add_argument("--runs", default="runs", help="runs/<kind>/<case>/<system>/")
    v.add_argument("--system", required=True, help="<agent>-<model>-<effort>, a run directory's name")
    v.add_argument("--out", required=True)
    v.add_argument("--frames", type=int, default=16)
    v.add_argument("--width", type=int, default=768)
    v.add_argument("--max-tokens", type=int, default=4096)
    v.add_argument("--no-layer1", action="store_true",
                   help="judge every render, including the blank ones")
    v.set_defaults(fn=cmd_vqa)

    w = sub.add_parser("pairwise", help="which of two renders is closer to the reference")
    common(w)
    w.add_argument("--cases", default="cases", help="cases/<kind>/<case>/reference.mp4")
    w.add_argument("--runs", default="runs", help="runs/<kind>/<case>/<system>/")
    w.add_argument("--out", required=True)
    w.add_argument("--systems", default="", help="comma-separated; default is every system under --runs")
    w.add_argument("--pairs", type=int, default=3,
                   help="pairs bought per scene, out of its round of the round-robin; 0 is all")
    # the reasoning trace can use up a smaller budget before the answer; the released
    # sweep allowed 256
    w.add_argument("--max-tokens", type=int, default=1024)
    w.add_argument("--fps", type=float, default=3.0)
    w.add_argument("--width", type=int, default=512)
    w.add_argument("--seed", type=int, default=0)
    w.set_defaults(fn=cmd_pairwise)

    s = sub.add_parser("score", help="VQA accuracy per system")
    s.add_argument("answers")
    s.add_argument("--questions", default=str(DATA))
    s.add_argument("--no-presence", action="store_true",
                   help="score every answered question, ignoring the presence gate")
    s.set_defaults(fn=cmd_score)

    e = sub.add_parser("elo", help="Bradley-Terry ratings from the pairwise file")
    e.add_argument("pairs")
    e.add_argument("--prior", type=float, default=1.0)
    e.add_argument("--boot", type=int, default=200,
                   help="bootstrap draws over tasks for the 95%% interval; 0 to skip")
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(fn=cmd_elo)

    a = ap.parse_args()
    if a.cmd in ("vqa", "pairwise"):
        # ffprobe counts the frames, so it is as required as ffmpeg; some builds ship
        # only one of the two
        for tool in ("ffmpeg", "ffprobe"):
            if not shutil.which(tool):
                sys.exit(f"{tool} is not on PATH")
    a.fn(a)


if __name__ == "__main__":
    main()
