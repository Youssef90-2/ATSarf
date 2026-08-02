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
    SESSION["graphs"][name] = {"persons": persons}

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
    SESSION["graphs"][out_name] = {"persons": persons}

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
):
    from biography_segmenter import BiographySegmenter
    if file is not None:
        raw = await file.read(); book_text = raw.decode("utf-8", errors="replace")
    elif text:
        book_text = text
    else:
        raise HTTPException(400, "provide a file or text")

    entries, clean_text, stats = BiographySegmenter().segment(book_text)
    ed = [e.to_dict() for e in entries]
    SESSION["biographies"][name] = ed

    rich = [e for e in ed if e["teachers"] or e["students"]]
    return {
        "name": name,
        "stats": {"entries": len(ed), "with_relations": len(rich),
                  "cross_refs": sum(1 for e in ed if e.get("is_reference"))},
        "entries": ed,
    }


# ===========================================================================
# STEP 5 — link biography entries to a graph (Table-2 boundary experiment)
# ===========================================================================
@app.post("/link-biography")
def link_biography(payload: dict = Body(...)):
    from biography_linker import Graph
    from kreachable import ReachabilityIndex
    from biography_linker import canonize_string
    from equality import narrator_distance

    bio_name = payload.get("bio_name", "khoei")
    graph_name = payload.get("graph_name")
    k = int(payload.get("k", 2))
    threshold = float(payload.get("threshold", 0.1))

    if bio_name not in SESSION["biographies"]:
        raise HTTPException(404, f"no biography '{bio_name}'")
    if graph_name not in SESSION["graphs"]:
        raise HTTPException(404, f"no graph '{graph_name}'")

    entries = SESSION["biographies"][bio_name]
    graph = Graph(SESSION["graphs"][graph_name]["persons"])
    reach = ReachabilityIndex(graph.persons, undirected=True)

    def match_ids(nm):
        return [pid for pid, _ in graph.match(nm, threshold)]

    gold = [e for e in entries if not e.get("is_reference") and match_ids(e["subject"])]
    denom = ng = wg = 0
    for e in gold:
        subj_ids = match_ids(e["subject"])
        # OLD-STYLE: use the full bag of co-mentioned narrators; fall back to
        # the labelled teachers/students if all_narrators is absent.
        neighbours = e.get("all_narrators") or (e["teachers"] + e["students"])
        for nb in neighbours:
            nb_ids = match_ids(nb)
            if not nb_ids:
                continue
            denom += 1
            if len(nb_ids) == 1:
                ng += 1
            if reach.any_reachable(subj_ids, nb_ids, k=k):
                wg += 1

    def ratio(a, b): return round(a / b, 3) if b else 0.0
    return {
        "bio": bio_name, "graph": graph_name, "k": k,
        "gold_entries": len(gold), "neighbours_in_graph": denom,
        "no_graph_recall": ratio(ng, denom),
        "with_graph_recall": ratio(wg, denom),
        "no_graph_correct": ng, "with_graph_correct": wg,
    }


# ===========================================================================
# session + health + index
# ===========================================================================
@app.get("/session")
def session():
    return {
        "books": {k: len(v) for k, v in SESSION["books"].items()},
        "graphs": {k: len(v["persons"]) for k, v in SESSION["graphs"].items()},
        "biographies": {k: len(v) for k, v in SESSION["biographies"].items()},
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
