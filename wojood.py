"""
wojood.py
=========
IDEA: Run the OFFICIAL Wojood NER model (SinaLab, Birzeit University)
without their `arabiner` package.

WHY THIS FILE EXISTS (document this in the report):
  * The official checkpoint on HuggingFace (SinaLab/ArabicNER-Wojood) is
    stored in arabiner's custom training format — NOT the standard
    transformers format — so `pipeline(...)` cannot load it.
  * The official `arabiner` package itself fails to install: its setup.py
    line 43 references an undefined variable `history` (a real bug).
  * Solution: we download the OFFICIAL weights and run them with our own
    thin loader that mirrors arabiner's architecture and inference logic
    exactly (verified by reading their source):
      - BertNestedTagger = AraBERTv2 + dropout + 21x Linear(768,4)
        (arabiner/nn/BertNestedTagger.py, BaseModel.py)
      - checkpoint loading: torch.load(...)["model"], DataParallel-prefixed
        (arabiner/trainers/BaseTrainer.py:104-117)
      - tokenization: per-word BertTokenizer subwords; the FIRST subword
        carries the word's tag (arabiner/data/transforms.py)
      - tag_vocab.pkl holds real torchtext Vocab objects: index 0 is the
        combined vocab, indexes 1: are one per entity type (21 types)
        (arabiner/utils/data.py + BertNestedTrainer.to_segments)

Same model, same weights, same logic — packaged cleanly.

USAGE:
    ner = WojoodNER()                      # downloads weights on first use
    flags = ner.person_flags(["حدثنا", "محمد", "بن", "يعقوب"])
    # -> [False, True, True, True]
"""

import pickle

import torch
import torch.nn as nn


# ===========================================================================
# 0. Standalone loader for tag_vocab.pkl
#    The pkl holds torchtext.vocab.Vocab objects, but torchtext's binary
#    extension is ABI-incompatible with modern torch (WinError 127 /
#    undefined symbol). We don't need torchtext at all — we only need the
#    ordered list of tag strings. This fake class captures the unpickled
#    state (a torch nn.Module-like object whose itos list we can recover)
#    without importing the real torchtext.
# ===========================================================================

class _FakeVocab:
    """Stand-in for torchtext.vocab.vocab.Vocab during unpickling."""
    def __setstate__(self, state):
        self._state = state

    def itos(self):
        """
        Recover the ordered token list. Verified structure of the SinaLab
        tag_vocab.pkl (torchtext Vocab as nn.Module):
            self._state['vocab']            -> nested _FakeVocab
            nested._state                   -> tuple (ver, [], TOKENS, [])
        So the token list is element index 2 of the inner vocab's state.
        We also keep fallbacks for other torchtext layouts.
        """
        st = getattr(self, "_state", {})

        # primary: state['vocab'] is a nested _FakeVocab whose state is a
        # tuple where one element is the list of token strings
        inner = st.get("vocab")
        if isinstance(inner, _FakeVocab):
            inner_state = getattr(inner, "_state", None)
            if isinstance(inner_state, (tuple, list)):
                # find the element that is a list of strings (tokens)
                best = []
                for part in inner_state:
                    if isinstance(part, list) and len(part) > len(best) \
                            and all(isinstance(x, str) for x in part):
                        best = part
                if best:
                    return best

        # fallbacks (older torchtext layouts)
        for key in ("itos_", "itos", "tokens"):
            if isinstance(st.get(key), list):
                return st[key]
        for val in st.values():
            if isinstance(val, list) and val and isinstance(val[0], str):
                return val
        return []


class _VocabUnpickler(pickle.Unpickler):
    """Redirect torchtext Vocab classes to our _FakeVocab."""
    def find_class(self, module, name):
        if "torchtext" in module and "Vocab" in name:
            return _FakeVocab
        if module == "torchtext.vocab.vocab" or name == "Vocab":
            return _FakeVocab
        try:
            return super().find_class(module, name)
        except Exception:
            # any other torchtext C++ helper -> harmless placeholder
            return _FakeVocab


def _load_tag_vocabs(path):
    """Return list of tag-string lists, one per entity type."""
    with open(path, "rb") as fh:
        raw = _VocabUnpickler(fh).load()
    return [v.itos() if isinstance(v, _FakeVocab) else list(v) for v in raw]


# ===========================================================================
# 1. The network — mirror of arabiner's BertNestedTagger
# ===========================================================================

class BertNestedTagger(nn.Module):

    def __init__(self, bert_config, num_labels):
        super().__init__()
        from transformers import BertModel
        # Build BERT from config only (random init) — real weights come
        # from the checkpoint, so we avoid downloading AraBERT twice.
        self.bert = BertModel(bert_config)
        self.dropout = nn.Dropout(0.1)
        self.classifiers = nn.Sequential(
            *[nn.Linear(768, n) for n in num_labels])
        self.max_num_labels = max(num_labels)

    def forward(self, input_ids, attention_mask):
        hidden = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask)["last_hidden_state"]
        hidden = self.dropout(hidden)
        outputs = []
        for classifier in self.classifiers:
            logits = classifier(hidden)
            pad = self.max_num_labels - logits.shape[-1]
            if pad:
                logits = nn.functional.pad(logits, (0, pad))
            outputs.append(logits)
        # B x T x L x C  (same layout as arabiner)
        return torch.stack(outputs).permute((1, 2, 0, 3))


# ===========================================================================
# 3. WojoodNER — download, load, and tag
# ===========================================================================

REPO_ID = "SinaLab/ArabicNER-Wojood"
BERT_NAME = "aubmindlab/bert-base-arabertv2"     # from the repo's args.json


class WojoodNER:

    def __init__(self, device: str = "cpu", batch_size: int = 8,
                 verbose: bool = True):
        from huggingface_hub import hf_hub_download
        from transformers import BertTokenizer, AutoConfig

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.verbose = verbose

        if verbose:
            print("[wojood] downloading/locating official files "
                  "(cached after first time)...")
        ckpt_path = hf_hub_download(REPO_ID, "checkpoints/checkpoint_0.pt")
        vocab_path = hf_hub_download(REPO_ID, "tag_vocab.pkl")

        # ---- tag vocabs: [0]=combined, [1:]=one per entity type (21) ----
        #    Loaded WITHOUT torchtext (ABI-incompatible). Each element is a
        #    plain list of tag strings; index in the list = the class id
        #    the model predicts for that entity-type head.
        all_vocabs = _load_tag_vocabs(vocab_path)
        self.type_vocabs = all_vocabs[1:]          # drop combined vocab

        if verbose:
            print(f"[wojood] recovered {len(self.type_vocabs)} entity types")
            # show first few tags of each type so we can confirm extraction
            preview = [t[:3] for t in self.type_vocabs[:5]]
            print("[wojood] sample type tags:", preview)

        # which head is PERS? (the person-name head we care about)
        self.pers_head = None
        for i, tags in enumerate(self.type_vocabs):
            if any(str(t).endswith("PERS") for t in tags):
                self.pers_head = i
                break
        if self.pers_head is None:
            raise RuntimeError(
                "PERS head not found. Recovered types: "
                f"{[t[:2] for t in self.type_vocabs]}")

        # ---- tokenizer + model ----
        self.tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
        config = AutoConfig.from_pretrained(BERT_NAME)
        num_labels = [len(v) for v in self.type_vocabs]
        model = BertNestedTagger(config, num_labels)

        checkpoint = torch.load(ckpt_path, map_location=self.device,
                                weights_only=False)
        state = checkpoint["model"]
        # strip DataParallel's "module." prefix (BaseTrainer wraps in DP)
        state = {k[len("module."):] if k.startswith("module.") else k: v
                 for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if verbose and (missing or unexpected):
            print(f"[wojood] state_dict: {len(missing)} missing, "
                  f"{len(unexpected)} unexpected keys (pooler etc. is fine)")

        model.eval().to(self.device)
        self.model = model
        if verbose:
            print("[wojood] ready. PERS head index:", self.pers_head)

    # ------------------------------------------------------------ chunking
    def _chunk_words(self, words, max_subwords=500):
        """
        Greedily pack words into segments that fit the 512-subword limit.
        Returns list of (word_indexes, subword_ids, first_subword_positions).
        """
        chunks = []
        cur_idx, cur_ids, cur_first = [], [], []
        for wi, word in enumerate(words):
            ids = self.tokenizer.encode(word, add_special_tokens=False) \
                  or [self.tokenizer.unk_token_id]
            if cur_ids and len(cur_ids) + len(ids) > max_subwords:
                chunks.append((cur_idx, cur_ids, cur_first))
                cur_idx, cur_ids, cur_first = [], [], []
            cur_first.append(len(cur_ids))        # position of 1st subword
            cur_ids.extend(ids)
            cur_idx.append(wi)
        if cur_idx:
            chunks.append((cur_idx, cur_ids, cur_first))
        return chunks

    # ------------------------------------------------------------- tagging
    @torch.no_grad()
    def person_flags(self, words):
        """
        The main API: list of words -> list of booleans
        (True = Wojood tags this word as part of a person name).
        Mirrors arabiner: prediction of a word = tag of its FIRST subword.
        """
        flags = [False] * len(words)
        chunks = self._chunk_words(words)
        cls_id, sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        pers_itos = self.type_vocabs[self.pers_head]   # plain list of tags

        for c_start in range(0, len(chunks), self.batch_size):
            batch = chunks[c_start:c_start + self.batch_size]
            max_len = max(len(ids) for _, ids, _ in batch) + 2
            input_ids, attention = [], []
            for _, ids, _ in batch:
                seq = [cls_id] + ids + [sep_id]
                pad = max_len - len(seq)
                input_ids.append(seq + [0] * pad)
                attention.append([1] * len(seq) + [0] * pad)

            input_ids = torch.tensor(input_ids, device=self.device)
            attention = torch.tensor(attention, device=self.device)
            logits = self.model(input_ids, attention)   # B x T x L x C
            preds = logits.argmax(dim=3)                # B x T x L

            for b, (word_idx, _, first_pos) in enumerate(batch):
                for wi, fp in zip(word_idx, first_pos):
                    tag_id = int(preds[b, fp + 1, self.pers_head])  # +1: CLS
                    if tag_id < len(pers_itos):
                        tag = pers_itos[tag_id]
                        if tag.endswith("-PERS"):       # B-PERS / I-PERS
                            flags[wi] = True

            if self.verbose and c_start % (self.batch_size * 10) == 0:
                done = min(c_start + self.batch_size, len(chunks))
                print(f"[wojood] segments {done}/{len(chunks)}", end="\r")

        if self.verbose:
            print()
        return flags


# ---------------------------------------------------------------------------
# Quick demo:  py -3.11 wojood.py    (downloads the model on first run!)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ner = WojoodNER()
    test_words = ["حدثنا", "محمد", "بن", "يعقوب", "عن", "علي", "بن",
                  "ابراهيم", "قال", "خلق", "الله", "العقل"]
    flags = ner.person_flags(test_words)
    print("\nword          PERS?")
    print("-" * 22)
    for w, f in zip(test_words, flags):
        print(f"{w:<12}  {'YES' if f else '-'}")
