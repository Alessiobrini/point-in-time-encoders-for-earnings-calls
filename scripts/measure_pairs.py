import numpy as np
from ecalls.train import pairs as PR, tokenizer as TK

df, _ = PR.load_pairs("qa-pairs-20241231")
s = df.sample(n=40000, seed=0)
tok = TK.load("ecalls-bpe-32768-20101231")
for col in ("text", "answer_text"):
    enc = tok.encode_batch(s[col].to_list(), add_special_tokens=False)
    L = np.array([len(e.ids) for e in enc])
    qs = np.percentile(L, [50, 90, 95, 99, 99.9])
    print("%-12s n=%d mean %.0f  p50 %.0f p90 %.0f p95 %.0f p99 %.0f p99.9 %.0f max %d"
          % (col, len(L), L.mean(), *qs, L.max()))
    for cap in (128, 192, 256, 320, 384, 512):
        print("   cap %4d: truncates %5.2f%% of texts, loses %5.2f%% of tokens"
              % (cap, 100 * (L > cap).mean(), 100 * np.clip(L - cap, 0, None).sum() / L.sum()))
