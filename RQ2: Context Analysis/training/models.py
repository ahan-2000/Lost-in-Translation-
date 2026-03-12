import torch, torch.nn as nn
from transformers import AutoModel

def mean_pool(H, mask):
    mask = mask.unsqueeze(-1).float()
    return (H*mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

# ---------------- Exp1: FieldsOnly ----------------
class FieldsOnlyModel(nn.Module):
    """
    Baseline: P(Y | OccitanWord, LatinLemma, LatinGender).
    Uses only the 'fields' encoder; no sentence context.
    """
    def __init__(self, pretrained="bert-base-multilingual-cased", hidden=256, dropout=0.2, freeze_encoder=True):
        super().__init__()
        self.fields_enc = AutoModel.from_pretrained(pretrained)
        if freeze_encoder:
            for p in self.fields_enc.parameters(): p.requires_grad = False
        d = self.fields_enc.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2)
        )

    def forward(self, input_ids_fields, attention_mask_fields, **kwargs):
        out_f = self.fields_enc(input_ids=input_ids_fields, attention_mask=attention_mask_fields, return_dict=True)
        pooled_f = mean_pool(out_f.last_hidden_state, attention_mask_fields)
        logits = self.mlp(pooled_f)
        return {"logits": logits}

# -------------- Exp2: Context + noun->sentence reader --------------
class ContextReaderModel(nn.Module):
    """
    P(Y | Sentence, i, LatinLemma, LatinGender, OccitanWord) with single-query cross-attention:
      Q = noun h_i (top layer), K,V = sentence layer (top or chosen lower layer).
    """
    def __init__(self, pretrained="bert-base-multilingual-cased", hidden=256, dropout=0.2, heads=4, dk=64, dv=64,
                 use_rel_bias=True, rel_window=32, layer_kv=-1, freeze_encoder=True, no_self_peek=False):
        super().__init__()
        self.sent_enc = AutoModel.from_pretrained(pretrained, output_hidden_states=True, return_dict=True)
        self.fields_enc = AutoModel.from_pretrained(pretrained)
        if freeze_encoder:
            for p in self.sent_enc.parameters(): p.requires_grad = False
            for p in self.fields_enc.parameters(): p.requires_grad = False
        self.d = self.sent_enc.config.hidden_size
        self.heads=heads; self.dk=dk; self.dv=dv
        self.use_rel_bias=use_rel_bias; self.rel_window=rel_window
        self.layer_kv=layer_kv; self.no_self_peek=no_self_peek

        self.WQ = nn.Parameter(torch.Tensor(heads, self.d, dk))
        self.WK = nn.Parameter(torch.Tensor(heads, self.d, dk))
        self.WV = nn.Parameter(torch.Tensor(heads, self.d, dv))
        nn.init.xavier_uniform_(self.WQ); nn.init.xavier_uniform_(self.WK); nn.init.xavier_uniform_(self.WV)
        if use_rel_bias:
            self.rel_bias = nn.Parameter(torch.zeros(heads, 2*rel_window+1))

        # classifier takes [h_i ; context_from_reader ; pooled_fields]
        self.proj = nn.Linear(self.d + heads*dv + self.d, hidden)
        self.mlp = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, input_ids_sent, attention_mask_sent, noun_pos,
                input_ids_fields, attention_mask_fields, return_attention=False, **kwargs):
        out_s = self.sent_enc(input_ids=input_ids_sent, attention_mask=attention_mask_sent)
        H_top = out_s.last_hidden_state
        layers = out_s.hidden_states
        if self.layer_kv >= 0:
            H_mem = layers[self.layer_kv]
        else:
            H_mem = H_top
        B,T,d = H_top.shape
        idx = noun_pos.view(B,1,1).expand(-1,1,d)
        h_i = H_top.gather(dim=1, index=idx).squeeze(1)

        # Q,K,V
        Q = torch.einsum("bd,hdk->bhk", h_i, self.WQ)       # (B,h,dk)
        K = torch.einsum("btd,hdk->bthk", H_mem, self.WK)   # (B,T,h,dk)
        V = torch.einsum("btd,hdv->bthv", H_mem, self.WV)   # (B,T,h,dv)
        scores = torch.einsum("bhk,bthk->bht", Q, K) / (self.dk**0.5)  # (B,h,T)

        # relative distance bias
        if self.use_rel_bias:
            device = scores.device
            pos = noun_pos.unsqueeze(-1)                    # (B,1)
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B,-1)
            delta = positions - pos
            w = self.rel_window
            delta = delta.clamp(-w, w) + w                  # [0,2w]
            rb = self.rel_bias[:, delta]                    # (h,B,T)
            rb = rb.permute(1,0,2)                          # (B,h,T)
            scores = scores + rb

        # forbid attending to noun token (optional)
        if self.no_self_peek:
            mask_self = torch.zeros_like(scores) - 1e9
            for b in range(B):
                mask_self[b, :, noun_pos[b]] = -1e9
            scores = scores + mask_self

        # padding mask
        am = attention_mask_sent.unsqueeze(1).expand(-1,self.heads,-1)
        scores = scores.masked_fill(am==0, -1e9)
        alpha = scores.softmax(dim=-1)                       # (B,h,T)
        c = torch.einsum("bht,bthv->bhv", alpha, V).reshape(B, self.heads*self.dv)

        # fields encoder
        out_f = self.fields_enc(input_ids=input_ids_fields, attention_mask=attention_mask_fields, return_dict=True)
        pooled_f = mean_pool(out_f.last_hidden_state, attention_mask_fields)

        r = torch.cat([h_i, c, pooled_f], dim=-1)
        logits = self.mlp(self.proj(r))
        
        result = {"logits": logits}
        if return_attention:
            result["attention_weights"] = alpha.detach().cpu()  # (B, heads, T)
            result["attention_scores"] = scores.detach().cpu()  # (B, heads, T)
            result["noun_positions"] = noun_pos.detach().cpu()  # (B,)
        
        return result

