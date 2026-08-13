"""
app.py — FastAPI backend for the Hadith system web UI (full pipeline)
=====================================================================
One `uvicorn app:app` run, then EVERYTHING happens from the web interface:

    /segment            upload a hadith book -> segment (sanad + matn)
    /build-graph        build a narrator graph from a segmented book
    /merge-graphs       merge several books' graphs into one
    /segment-biography  upload a rijal book  -> entries (+teachers/students)
    /link-biography     link entries to a graph -> Table-2 boundary result
    /session            list what's stored this session
    /                   serves the single-page interface (app.html)

Results are kept in an in-memory SESSION dict keyed by a name the user gives
(e.g. "kafi1"), so step N+1 can use step N's output without re-uploading.

Run:
    pip install fastapi uvicorn python-multipart
    py -3.11 -m uvicorn app:app --port 8000
then open  http://localhost:8000/
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from engine import ArabicEngine
from segmenter import HadithSegmenter, SegmenterConfig
from fsm import FsmParams
from models import Chain, Narrator, NameConnector, NamePrim
from graph_build import GraphParams
from narrator_graph import NarratorGraph
from normalization import normalize

# --- the biography stage: the SAME modules run_biography.py drives ----------
# There is exactly one biography method in this project, and it is the
# paper's: BiographyFSM (§4 automaton) -> GraphIndex (§4 isRealNarrator) ->
# BiographyDetector (§4 findChosenBiography / k-reachable boundaries).
# The web UI must not have a second, simpler one — a result you can only get
# from the UI is not a result you can put in the report.
from bio_fsm import BiographyFSM, BioParams
from biography_detector import BiographyDetector, BiographyParams
from narrator_matcher import GraphIndex

app = FastAPI(title="Hadith System API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# in-memory session store: name -> {"hadiths":[...], "graph":{...}, ...}
# ---------------------------------------------------------------------------
SESSION: dict = {"books": {}, "graphs": {}, "biographies": {}}

# ---------------------------------------------------------------------------
# lazy engine (Wojood is expensive; build once, reuse)
# ---------------------------------------------------------------------------
_ENGINE = {"engine": None, "use_wojood": True}
def get_engine(use_wojood=True) -> ArabicEngine:
    if _ENGINE["engine"] is None or _ENGINE["use_wojood"] != use_wojood:
        _ENGINE["engine"] = ArabicEngine(use_wojood=use_wojood)
        _ENGINE["use_wojood"] = use_wojood
    return _ENGINE["engine"]


# ===========================================================================
# STEP 1 — segmentation
# ===========================================================================
@app.post("/segment")
async def segment(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    name: str = Form("book"),
    narr_min: int = Form(3),
    start_marker: str = Form(""),
    use_wojood: bool = Form(True),
):
    if file is not None:
        raw = await file.read()
        book_text = raw.decode("utf-8", errors="replace")
    elif text:
        book_text = text
    else:
        raise HTTPException(400, "provide a file or text")

    engine = get_engine(use_wojood)
    cfg = SegmenterConfig(fsm=FsmParams(narr_min=narr_min),
                          start_marker=start_marker or "")
    seg = HadithSegmenter(engine, cfg)
    hadiths, stats, clean_text, _imap = seg.segment(book_text, hadith_mode=True)

    out = []
    for h in hadiths:
        d = h.to_dict()
        narrators = d["sanad"]["narrators"]
        s_start = d["sanad"].get("start", 0)
        m_end = (d["matn"]["end"] if isinstance(d["matn"], dict)
                 else s_start)
        original = clean_text[s_start:m_end].strip() if m_end > s_start else ""
        out.append({
            "number": d["number"],
            "narrator_count": len(narrators),
            "narrators": [n["name"] for n in narrators],
            "narrators_full": narrators,
            "matn": (d["matn"]["text"] if isinstance(d["matn"], dict)
                     else d["matn"]),
            "original": original,        # raw text of this hadith (show)
            "suspicious": d["number"] == -1 or len(narrators) < narr_min,
        })

    # store the FULL hadith dicts (with positions) for graph building later
    SESSION["books"][name] = [h.to_dict() for h in hadiths]

    clean = [h for h in out if not h["suspicious"]]
    return JSONResponse({
        "name": name,
        "book_preview": clean_text[:3000],      # show raw book text
        "book_chars": len(clean_text),
        "stats": {
            "total_hadiths": len(out),
            "clean": len(clean),
            "suspicious": len(out) - len(clean),
            "wojood": getattr(engine, "wojood", None) and engine.wojood.available,
        },
        "hadiths": out,
    })


# ===========================================================================
# STEP 2 — build a narrator graph from a stored (segmented) book
# ===========================================================================
def _chains_from_book(hadiths):
    """Rebuild Chain objects from stored hadith dicts (reuse run_graph logic)."""
    from run_graph import narrator_from_dict
    chains = []
    for h in hadiths:
        chain = Chain()
        for i, nd in enumerate(h["sanad"]["narrators"]):
            if i:
                chain.add(NameConnector("عن", 0, 0))
            chain.add(narrator_from_dict(nd))
        chains.append(chain)
    return chains


@app.post("/build-graph")
def build_graph(payload: dict = Body(...)):
    name = payload.get("name", "book")
    threshold = float(payload.get("threshold", 0.5))
    clean_only = bool(payload.get("clean_only", True))

    if name not in SESSION["books"]:
        raise HTTPException(404, f"no segmented book named '{name}'")
    hadiths = SESSION["books"][name]
    if clean_only:
        hadiths = [h for h in hadiths if h["number"] != -1
                   and len(h["sanad"]["narrators"]) >= 3]

    chains = _chains_from_book(hadiths)
    params = GraphParams(equality_threshold=threshold, equality_radius=3,
                         qualifiers_mode="off")
    graph = NarratorGraph(params, break_cycles=True, delta=0.2)
    graph.build(chains)

    persons = [p.to_dict() for p in graph.persons.values()]
    # Store `params` too: GraphIndex reads equality_threshold from here so the
    # biography stage matches at the threshold the graph was BUILT at (D7).
    # Without it every lookup silently falls back to a hardcoded default.
    SESSION["graphs"][name] = {
        "persons": persons,
        "params": {"equality_threshold": threshold, "equality_radius": 3,
                   "qualifiers_mode": "off"},
    }

    edges = sum(len(p["children"]) for p in persons)
    top = sorted(persons, key=lambda p: p["occurrences"], reverse=True)[:10]
    return {
        "name": name,
        "stats": {"persons": len(persons), "edges": edges,
                  "merged": sum(1 for p in persons if p["occurrences"] > 1),
                  "max_rank": max((p["rank"] for p in persons), default=0)},
        "top": [{"name": p["primary_name"], "occurrences": p["occurrences"],
                 "children": len(p["children"])} for p in top],
        "persons": persons,
    }


# ===========================================================================
# STEP 3 — merge several stored graphs' books into one graph
# ===========================================================================
@app.post("/merge-graphs")
def merge_graphs_ep(payload: dict = Body(...)):
    names = payload.get("names", [])
    threshold = float(payload.get("threshold", 0.5))
    out_name = payload.get("out_name", "merged")
    missing = [n for n in names if n not in SESSION["books"]]
    if missing:
        raise HTTPException(404, f"not segmented: {missing}")
    if len(names) < 2:
        raise HTTPException(400, "give at least two book names")

    all_chains, provenance = [], []
    for n in names:
        hadiths = [h for h in SESSION["books"][n]
                   if h["number"] != -1 and len(h["sanad"]["narrators"]) >= 3]
        chains = _chains_from_book(hadiths)
        all_chains.extend(chains)
        provenance.extend([n] * len(chains))

    params = GraphParams(equality_threshold=threshold, equality_radius=3,
                         qualifiers_mode="off")
    graph = NarratorGraph(params, break_cycles=True, delta=0.2)
    graph.build(all_chains)

    # book provenance per person could be added here later
    persons = [p.to_dict() for p in graph.persons.values()]
    SESSION["graphs"][out_name] = {
        "persons": persons,
        "params": {"equality_threshold": threshold, "equality_radius": 3,
                   "qualifiers_mode": "off"},
    }

    edges = sum(len(p["children"]) for p in persons)
    return {
        "name": out_name,
        "merged_books": names,
        "stats": {"persons": len(persons), "edges": edges,
                  "merged": sum(1 for p in persons if p["occurrences"] > 1)},
        "persons": persons,
    }


# ===========================================================================
# STEP 4 — segment a rijal (biography) book
# ===========================================================================
@app.post("/segment-biography")
async def segment_biography(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    name: str = Form("khoei"),
    use_wojood: bool = Form(True),
):
    if file is not None:
        raw = await file.read(); book_text = raw.decode("utf-8", errors="replace")
    elif text:
        book_text = text
    else:
        raise HTTPException(400, "provide a file or text")

    # Paper §4: "ANGE computes lb, the list of detected narrators in the
    # biography books using similar techniques used in the hadith books, and
    # orders lb by the order of appearance."
    #
    # This step is the NO-GRAPH baseline — BiographyFSM with confirm=None,
    # exactly the paper's control condition. Entry BOUNDARIES are not decided
    # here; they come from the graph in step 5. That separation is the whole
    # experiment, so it must not be short-circuited by guessing entries from
    # a "N - name:" regex.
    clean_text, _index_map = normalize(book_text)
    engine = get_engine(use_wojood)
    tokens = engine.analyze_cached(clean_text)

    fsm = BiographyFSM(BioParams(), confirm=None)
    mentions = fsm.run(tokens)

    # Keep the tokens: step 5 re-runs the automaton WITH the graph, and
    # re-analysing a whole rijal book through CAMeL+Wojood would cost minutes.
    SESSION["biographies"][name] = {
        "clean_text": clean_text,
        "tokens": tokens,
        "mentions": [m.to_dict() for m in mentions],
        "fsm_stats": dict(fsm.stats),
    }

    return {
        "name": name,
        "stats": {
            "chars": len(clean_text),
            "tokens": len(tokens),
            "mentions": len(mentions),
            "narrator_runs": len(getattr(fsm, "candidates", [])),
            "runs_ended_by_budget": fsm.stats.get("runs_ended_by_budget", 0),
            "wojood": getattr(engine, "wojood", None) and engine.wojood.available,
        },
        "note": "قائمة الرواة (lb) بدون الشبكة — الحدود تُحدَّد في الخطوة ٥",
        "mentions": [
            {"name": m.name, "start": m.start, "end": m.end,
             "context": clean_text[max(0, m.start - 40):m.end + 40]}
            for m in mentions[:60]
        ],
    }


# ===========================================================================
# STEP 5 — link biography entries to a graph (Table-2 boundary experiment)
# ===========================================================================
@app.post("/link-biography")
def link_biography(payload: dict = Body(...)):
    """
    The paper's cross-document experiment (§4, Table 2), run exactly as
    run_biography.py runs it: the SAME automaton twice, differing in ONE
    argument.

        NO-GRAPH    BiographyFSM(confirm=None)
        WITH-GRAPH  BiographyFSM(confirm=GraphIndex.confirm)

    That single switch is the independent variable. The FSM, its parameters,
    the detector and the k-reachable radius are identical across the two, which
    is what makes the columns comparable.

    NOTE ON WHAT THIS DOES *NOT* RETURN: recall/precision against gold. Those
    are Table 2 proper and need a reviewed gold file — run_biography.py --gold.
    What comes back here is the pipeline's own behaviour, which is observable
    without gold and is where the mechanism is visible: the graph does not find
    MORE narrators, it stops the runs from dying.
    """
    bio_name = payload.get("bio_name", "khoei")
    graph_name = payload.get("graph_name")
    k = int(payload.get("k", 1))                      # old code hardcodes 1
    near = int(payload.get("near", 100))              # old bio_nrc_max
    bio_threshold = float(payload.get("bio_threshold", 2.0))
    sequential = bool(payload.get("sequential", False))

    if bio_name not in SESSION["biographies"]:
        raise HTTPException(404, f"no biography '{bio_name}'")
    if graph_name not in SESSION["graphs"]:
        raise HTTPException(404, f"no graph '{graph_name}'")

    stored = SESSION["biographies"][bio_name]
    tokens = stored["tokens"]
    clean_text = stored["clean_text"]

    index = GraphIndex(SESSION["graphs"][graph_name])
    det_params = BiographyParams(near_max_chars=near, threshold=bio_threshold,
                                 k=k, sequential_disambiguation=sequential)

    def condition(confirm):
        fsm = BiographyFSM(BioParams(), confirm=confirm)
        mentions = fsm.run(tokens)
        detector = (BiographyDetector(mentions, index, det_params)
                    if confirm is not None else None)
        detected = detector.detect() if detector else []
        return {
            "mentions": len(mentions),
            "confirmed_mentions": sum(1 for m in mentions if m.confirmed),
            "narrator_runs": len(getattr(fsm, "candidates", [])),
            "runs_ended_by_budget": fsm.stats.get("runs_ended_by_budget", 0),
            "entries_detected": len(detected),
            "avg_entry_chars": (round(sum(b.end - b.start for b in detected)
                                      / len(detected)) if detected else 0),
        }, detected, detector

    no_graph, _, _ = condition(None)
    with_graph, detected, detector = condition(index.confirm)

    return {
        "bio": bio_name,
        "graph": graph_name,
        "params": {"k": k, "near_max_chars": near,
                   "bio_threshold": bio_threshold,
                   "sequential_disambiguation": sequential,
                   "match_threshold": index.threshold},
        "graph_stats": index.stats(),
        "no_graph": no_graph,
        "with_graph": with_graph,
        "disambiguation": detector.disambiguation_stats if detector else {},
        "entries": [
            dict(b.to_dict(),
                 text=clean_text[b.start:min(b.end, b.start + 300)])
            for b in detected[:40]
        ],
        "reportable": False,
        "note": ("أرقام السلوك فقط. Table 2 (recall/precision) تحتاج ملف gold "
                 "مُراجَع عبر run_biography.py --gold"),
    }


# ===========================================================================
# session + health + index
# ===========================================================================
@app.get("/session")
def session():
    return {
        "books": {k: len(v) for k, v in SESSION["books"].items()},
        "graphs": {k: len(v["persons"]) for k, v in SESSION["graphs"].items()},
        "biographies": {k: len(v["mentions"])
                        for k, v in SESSION["biographies"].items()},
    }

@app.get("/health")
def health():
    eng = _ENGINE["engine"]
    return {"status": "ok", "engine_loaded": eng is not None}

@app.get("/", response_class=HTMLResponse)
def index():
    p = Path(__file__).parent / "web" / "app.html"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return "<h1>app.html not found in web/</h1>"
