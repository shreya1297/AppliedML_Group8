import re
from collections import Counter

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z0-9]")

def basic_tokenize(text: str):
    return TOKEN_RE.findall(str(text).lower())

class TrainOnlyVocab:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos

    @property
    def pad_id(self): return self.stoi["[PAD]"]
    @property
    def unk_id(self): return self.stoi["[UNK]"]
    @property
    def cls_id(self): return self.stoi["[CLS]"]
    @property
    def sep_id(self): return self.stoi["[SEP]"]
    @property
    def vocab_size(self): return len(self.itos)

    @classmethod
    def build(cls, df, max_vocab=30000, min_freq=2):
        """
        Build vocab from TRAIN ONLY:
        context + question + all options
        df: pandas DataFrame where answers is a list of 4 strings
        """
        ctr = Counter()
        for _, r in df.iterrows():
            ctr.update(basic_tokenize(r["context"]))
            ctr.update(basic_tokenize(r["question"]))
            for a in r["answers"]:
                ctr.update(basic_tokenize(a))

        itos = list(SPECIALS)
        for tok, f in ctr.most_common():
            if f < min_freq:
                break
            if tok in SPECIALS:
                continue
            itos.append(tok)
            if len(itos) >= max_vocab:
                break
        stoi = {t: i for i, t in enumerate(itos)}
        return cls(stoi, itos)

    def encode_tokens(self, tokens):
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def encode_text(self, text: str):
        return self.encode_tokens(basic_tokenize(text))
