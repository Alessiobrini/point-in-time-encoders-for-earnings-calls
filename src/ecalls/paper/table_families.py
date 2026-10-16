"""Table 2: the retrievers and encoders compared, with sizes measured and token counts from run records.

    python -m ecalls.paper.table_families

Sizes come from `reports/paper/model_facts.json` (the encoder body as the benchmark loads it). Our
token counts come from the run records: 1.2B per from-scratch vintage (D21), the chain's cumulative
count from `seed_summary.chain_tokens`, 2.4B for the twice-budget vintage. ChronoBERT's come from
He, Lv, Manela and Wu (JFE 2026): 460B for the 1999 base (p. 3) and 525B for the 2024 vintage (p. 6).
FinBERT's corpus is 4.9B tokens (Yang, Uy and Huang, arXiv 2006.08097, Table 1); how many it read in
pretraining is not stated, so the table gives the corpus size and says so.
"""
import json

from ecalls.diagnostics import seed_summary as S
from ecalls.paper import common as C

FACTS = C.P.REPORTS_DIR / "paper" / "model_facts.json"
OUT = C.TAB_DIR / "families.tex"
CHRONO_TOKENS_B = (460, 525)          # JFE 2026, pp. 3 and 6


def size(facts: dict, model: str) -> tuple[str, str]:
    f = facts[model]
    return f"{f['params'] / 1e6:.1f}", str(f["dim"])


def rows(facts: dict, chain: dict[int, int]) -> list[list[str] | str]:
    lo, hi = chain[2010] / 1e9, chain[2024] / 1e9
    out: list[list[str] | str] = [r"\multicolumn{8}{l}{\emph{References}}"]
    out.append(["BM25", "n/a", "n/a", "n/a", "n/a", "word counts", "none", "n/a"])
    for name, m in (("bge-small", "bge-small-en-v1.5"), ("bge-base", "bge-base-en-v1.5"),
                    ("bge-large", "bge-large-en-v1.5"), ("e5-base", "e5-base-v2")):
        p, d = size(facts, m)
        out.append([name, p, d, "general web", "not stated", "WordPiece", "general pairs", "unknown"])
    p, d = size(facts, "finbert-pretrain")
    out.append(["FinBERT", p, d, "filings, calls, reports", r"4.9B corpus", "FinVocab",
                "none", "corpus to 2019"])
    p, d = size(facts, "chrono-bert-v1-20241231")
    out.append(["ChronoBERT", p, d, "general", f"{CHRONO_TOKENS_B[0]}B to {CHRONO_TOKENS_B[1]}B",
                "BPE, ModernBERT", "none", "2010 to 2024"])
    out.append(r"\multicolumn{8}{l}{\emph{Contrastive fine-tuning on call question-answer pairs}}")
    out.append(["ChronoBERT + contrastive fine-tuning", p, d, "general",
                f"{CHRONO_TOKENS_B[0]}B to {CHRONO_TOKENS_B[1]}B", "BPE, ModernBERT", "call pairs", "2010 to 2024"])
    out.append(["ChronoBERT + DAPT", p, d, "general, then calls", r"+1.2B on calls", "BPE, ModernBERT",
                "call pairs", "2010, 2017, 2024"])
    p, d = size(facts, "ecalls-ret-20241231")
    out.append(["Ours, from scratch", p, d, "calls", "1.2B", "BPE, 2010 calls", "call pairs", "2010 to 2024"])
    out.append(["Ours, chained", p, d, "calls", f"{lo:.1f}B to {hi:.1f}B", "BPE, 2010 calls", "call pairs",
                "2010 to 2024"])
    out.append(r"\multicolumn{8}{l}{\emph{Controls for training length}}")
    out.append(["Ours, twice the tokens", p, d, "calls", "2.4B", "BPE, 2010 calls", "call pairs", "2024"])
    # replay Y reads as many tokens as chained segment Y+1, for Y from 2010 to 2023
    seg = [(chain[y + 1] - chain[y]) / 1e9 for y in range(2010, 2024)]
    out.append(["Ours, replay control", p, d, "calls", f"{min(seg):.2f}B to {max(seg):.2f}B", "BPE, 2010 calls",
                "call pairs", "2010 to 2023"])
    return out


def render(body: list[list[str] | str]) -> str:
    head = ("Model", "Params (M)", "Dim.", "Pretraining text", "Tokens", "Tokenizer", "Fine-tuning", "Cutoff")
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Retrievers and encoders compared. Parameters count the encoder body as loaded for the "
             r"benchmark, and Dim.\ is the embedding dimension. Tokens are the tokens read in "
             r"pretraining, and every from-scratch vintage reads the same 1.2B whatever the size of its window. For our yearly families the range runs from the 2010 vintage to the 2024 one, and for ChronoBERT from its 1999 base, where its chain starts, to its 2024 vintage. "
             r"FinBERT's entry is the size of its corpus, since the tokens it read are not stated. Contrastive "
             r"fine-tuning trains on question-answer pairs from trainable calls dated on or before each "
             r"vintage's cutoff. The replay control continues chained vintage $Y$ on its own last year for as "
             r"many tokens as the next segment reads.}",
             r"\label{tab:families}",
             r"\begin{tabular}{l r r l l l l l}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for r in body:
        lines.append(r + r" \\" if isinstance(r, str) else " & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def main() -> int:
    facts = json.loads(FACTS.read_text())["models"]
    tex = render(rows(facts, S.chain_tokens()))
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
