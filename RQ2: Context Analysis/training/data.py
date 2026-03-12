import pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, Any

def normalize_label(v):
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ["f","fem","female","feminine"]: return 1
        if s in ["m","masc","male","masculine"]: return 0
    return int(v)

def build_fields_text(occ_word: str, latin_lemma: str, latin_gender: str,
                      exclude_latin: bool = False) -> str:
    occ = str(occ_word) if isinstance(occ_word, str) else ""
    if exclude_latin:
        # Only the Occitan word — no Latin lemma or Latin gender
        return f"[OCC]{occ}"
    lg = str(latin_gender).strip().upper()
    if lg in ["F","FEM","FEMININE"]: lg_tok = "LATGENDER_F"
    elif lg in ["M","MASC","MASCULINE"]: lg_tok = "LATGENDER_M"
    elif lg in ["N","NEUT","NEUTER"]: lg_tok = "LATGENDER_N"
    else: lg_tok = f"LATGENDER_{lg}"
    lem = str(latin_lemma) if isinstance(latin_lemma, str) else ""
    # compact fields string (no sentence)
    return f"[OCC]{occ} [LAT]{lem} [LG]{lg_tok}"

class OccitanDataset(Dataset):
    """
    Always stores BOTH encodings:
      - sentence (masked or not)
      - fields (Occitan word + Latin lemma + Latin gender)
    And returns noun_pos (subword index in the *used* sentence), the gold label, and row_id.
    """
    def __init__(self, df: pd.DataFrame, sent_tok: AutoTokenizer, fields_tok: AutoTokenizer,
                 sentence_col: str, noun_index_col: str, label_col: str,
                 occitan_word_col: str, latin_lemma_col: str, latin_gender_col: str,
                 max_len_sent: int = 128, max_len_fields: int = 32,
                 index_in_whitespace: bool = True, mask_noun: bool = False, masked_token: str = "NOUNTOKEN",
                 exclude_latin: bool = False):
        self.df = df.reset_index().rename(columns={"index":"row_id"})
        self.sent_tok = sent_tok
        self.fields_tok = fields_tok
        self.sentence_col = sentence_col
        self.noun_index_col = noun_index_col
        self.label_col = label_col
        self.occ_col = occitan_word_col
        self.lat_col = latin_lemma_col
        self.latg_col = latin_gender_col
        self.max_len_sent = max_len_sent
        self.max_len_fields = max_len_fields
        self.index_in_whitespace = index_in_whitespace
        self.mask_noun = mask_noun
        self.masked_token = masked_token
        self.exclude_latin = exclude_latin

        self.labels = self.df[self.label_col].map(normalize_label).astype(int).tolist()
        self.cached = []

        for row_ix, row in self.df.iterrows():
            sent_orig = str(row[self.sentence_col])
            idx_ws = int(row[self.noun_index_col])
            occ = row[self.occ_col]
            lat = row[self.lat_col]
            latg = row[self.latg_col]

            # choose sentence to encode (masked or not)
            if self.mask_noun:
                ws = sent_orig.split()
                if 0 <= idx_ws < len(ws):
                    ws[idx_ws] = self.masked_token
                sent_used = " ".join(ws)
            else:
                sent_used = sent_orig

            # sentence encoding
            enc_sent = self.sent_tok(
                sent_used, max_length=self.max_len_sent,
                truncation=True, padding="max_length",
                return_offsets_mapping=True
            )

            # map whitespace index -> first overlapping subword IN THE USED SENTENCE
            noun_pos = 1
            if self.index_in_whitespace:
                ws_toks = sent_used.split()
                # rebuild char spans for sent_used
                spans = []
                start = 0
                for tok in ws_toks:
                    s = sent_used.find(tok, start)
                    spans.append((s, s+len(tok)))
                    start = s+len(tok)
                if 0 <= idx_ws < len(spans):
                    target_span = spans[idx_ws]
                    for j,(a,b) in enumerate(enc_sent["offset_mapping"]):
                        if a==b: continue
                        if a < target_span[1] and b > target_span[0]:
                            noun_pos = j; break
            else:
                noun_pos = max(1, min(len(enc_sent["input_ids"])-2, idx_ws))

            # fields encoding (Occitan word + optionally Latin lemma + Latin gender)
            fields_text = build_fields_text(occ, lat, latg, exclude_latin=self.exclude_latin)
            enc_fields = self.fields_tok(
                fields_text, max_length=self.max_len_fields,
                truncation=True, padding="max_length"
            )

            pack = {
                "row_id": torch.tensor(int(row["row_id"])).long(),
                "input_ids_sent": torch.tensor(enc_sent["input_ids"]).long(),
                "attention_mask_sent": torch.tensor(enc_sent["attention_mask"]).long(),
                "noun_pos": torch.tensor(noun_pos).long(),
                "input_ids_fields": torch.tensor(enc_fields["input_ids"]).long(),
                "attention_mask_fields": torch.tensor(enc_fields["attention_mask"]).long(),
                "label": torch.tensor(self.labels[row_ix]).long(),
            }
            if "token_type_ids" in enc_sent:
                pack["token_type_ids_sent"] = torch.tensor(enc_sent["token_type_ids"]).long()
            if "token_type_ids" in enc_fields:
                pack["token_type_ids_fields"] = torch.tensor(enc_fields["token_type_ids"]).long()
            self.cached.append(pack)

    def __len__(self): return len(self.cached)
    def __getitem__(self, ix: int) -> Dict[str, Any]:
        return self.cached[ix]

def build_loader(df, sent_tok, fields_tok,
                 sentence_col, noun_index_col, label_col,
                 occitan_word_col, latin_lemma_col, latin_gender_col,
                 max_len_sent=128, max_len_fields=32, batch_size=16, shuffle=False,
                 index_in_whitespace=True, mask_noun=False, masked_token="NOUNTOKEN",
                 exclude_latin=False):
    ds = OccitanDataset(df, sent_tok, fields_tok, sentence_col, noun_index_col, label_col,
                        occitan_word_col, latin_lemma_col, latin_gender_col,
                        max_len_sent, max_len_fields, index_in_whitespace, mask_noun, masked_token,
                        exclude_latin=exclude_latin)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

