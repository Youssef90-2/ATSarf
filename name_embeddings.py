"""
name_embeddings.py
==================
IDEA (our modern addition — NOT in the paper or old code):
Compare narrator names by MEANING/FORM using neural embeddings, not just
by hierarchical structure. Each name becomes a vector (a numeric
fingerprint); two names' similarity = cosine of their vectors.

WHY (motivation for the report):
The hierarchical metric (equality.py) compares name words LITERALLY.
It is precise (it hard-rejects a different father) but brittle to:
    - spelling variants across editions (علي / علا, الحسين / الحسن...)
    - the same name written slightly differently
Embeddings capture that two forms are "close" even when characters differ.

IMPORTANT DESIGN DECISION:
Embeddings are a COMPLEMENT, not a replacement. The hierarchical metric's
hard-reject (different father -> 0) must stay — otherwise we repeat the
edit-distance mistake of fusing different people with similar spelling.
So equality.py uses embeddings only to *boost* an already-plausible match,
never to override a hard reject. (See equality.py metric_mode="hybrid".)

TOOL: AraBERT v2 (aubmindlab/bert-base-arabertv2) — the SAME model Wojood
is built on, already in your HuggingFace cache, so no new large download.

PERFORMANCE: BERT is slow. Two safeguards:
    - embed only UNIQUE names (names repeat thousands of times), then cache
    - a disk cache keyed by name, so re-runs are instant
GRACEFUL DEGRADATION: if AraBERT can't load (offline/missing), .available
is False and the system falls back to the hierarchical metric alone.
"""

import hashlib
import json
import os


ARABERT_MODEL = "aubmindlab/bert-base-arabertv2"


class NameEmbeddings:
    """Turns names into vectors and scores their similarity."""

    def __init__(self, enabled: bool = True, cache_dir: str = ".emb_cache",
                 verbose: bool = True):
        self.available = False
        self.model = None
        self.tokenizer = None
        self.verbose = verbose
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._mem_cache = {}          # name -> vector (in-memory, this run)

        if not enabled:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            self._torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(ARABERT_MODEL)
            self.model = AutoModel.from_pretrained(ARABERT_MODEL)
            self.model.eval()
            self.available = True
            if verbose:
                print("[embeddings] AraBERT loaded.")
        except Exception as error:
            if verbose:
                print(f"[embeddings] not available "
                      f"({type(error).__name__}) -> hierarchical only.")

    # ------------------------------------------------------------- embed
    def embed(self, name: str):
        """
        Return the embedding vector for a name (list of floats), or None
        if embeddings are unavailable. Uses mean-pooling over token vectors
        (standard, robust sentence/phrase embedding).
        """
        if not self.available or not name:
            return None
        if name in self._mem_cache:
            return self._mem_cache[name]

        # disk cache
        cached = self._load_disk(name)
        if cached is not None:
            self._mem_cache[name] = cached
            return cached

        torch = self._torch
        with torch.no_grad():
            inputs = self.tokenizer(name, return_tensors="pt",
                                    truncation=True, max_length=32)
            outputs = self.model(**inputs)
            # mean-pool the last hidden state over real tokens
            hidden = outputs.last_hidden_state[0]         # (tokens, 768)
            mask = inputs["attention_mask"][0].unsqueeze(-1)
            summed = (hidden * mask).sum(dim=0)
            count = mask.sum().clamp(min=1)
            vector = (summed / count).tolist()

        self._mem_cache[name] = vector
        self._save_disk(name, vector)
        return vector

    # ------------------------------------------------------------- similarity
    def similarity(self, name1: str, name2: str):
        """
        Cosine similarity in [0,1] (0.5-shifted so it stays non-negative),
        or None if embeddings unavailable.
        """
        v1, v2 = self.embed(name1), self.embed(name2)
        if v1 is None or v2 is None:
            return None
        return self._cosine(v1, v2)

    @staticmethod
    def _cosine(v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = sum(a * a for a in v1) ** 0.5
        n2 = sum(b * b for b in v2) ** 0.5
        if n1 == 0 or n2 == 0:
            return 0.0
        cos = dot / (n1 * n2)                 # in [-1, 1]
        return max(0.0, min(1.0, (cos + 1) / 2))   # map to [0, 1]

    # ------------------------------------------------------------- disk cache
    def _key_path(self, name):
        key = hashlib.md5(name.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, key + ".json")

    def _load_disk(self, name):
        path = self._key_path(name)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_disk(self, name, vector):
        try:
            with open(self._key_path(name), "w", encoding="utf-8") as f:
                json.dump(vector, f)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Quick demo:  py -3.11 name_embeddings.py   (loads AraBERT, first time slow)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    emb = NameEmbeddings()
    if not emb.available:
        print("AraBERT unavailable here — this demo needs the model.")
    else:
        pairs = [
            ("علي بن ابراهيم", "علا بن ابراهيم"),   # spelling variant (ي/ا)
            ("محمد بن يعقوب", "محمد بن يعقوب الكليني"),  # same + nisba
            ("محمد بن يعقوب", "محمد بن الحسن"),      # different father
            ("الحسين بن محمد", "الحسن بن محمد"),     # close but different
            ("جعفر", "قتيبه"),                       # unrelated
        ]
        print(f"{'name A':<20}{'name B':<24}{'cosine':>8}")
        print("-" * 54)
        for a, b in pairs:
            s = emb.similarity(a, b)
            print(f"{a:<20}{b:<24}{s:>8.3f}")
        print("\nNote: embeddings give HIGH scores even to different people "
              "with similar spelling (محمد بن يعقوب / محمد بن الحسن) — which "
              "is exactly why they must only BOOST the hierarchical metric, "
              "never override its hard-reject.")
