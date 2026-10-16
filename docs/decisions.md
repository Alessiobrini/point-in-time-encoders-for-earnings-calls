# Design decisions cited in the code

Comments in the code refer to design decisions by number (D1, D29, ...). This index gives each one in a
sentence. The paper states the ones its results rest on.

| | Decision |
|---|---|
| D1 | Point-in-time S&P 500 membership comes from `crsp.dsp500list_v2`, because Compustat's index history (`comp.idxcst_his`, `gvkeyx='000003'`) holds current members only. |
| D2 | Index membership is a label joined onto the corpus, never a filter at pull time. |
| D3 | Membership past the CRSP horizon is carried forward, and removals are not truncated. |
| D4 | Of the several Capital IQ versions of a call, the one with the most text is kept, measured rather than ranked by collection type. |
| D5 | Transcript length is aggregated on the WRDS server, so version selection reads no text. |
| D6 | The corpus layer is archival, and normalization of the text is minimal. |
| D7 | Speaker turns are stored in shards by calendar quarter of the call date. |
| D9 | A refresh pulls transcripts created after a watermark with a safety margin, and reconciles counts. |
| D11 | Each vintage is trained independently and selected at inference from a registry by date. |
| D12 | Early interim embeddings came from ChronoBERT's 2024 vintage. |
| D13 | Embeddings are stored in float16, and the forward pass runs in float16. |
| D16 | Retrieval and prediction use separate embedding sets, never the same one for both. |
| D17 | Retrieval is hybrid, BM25 plus dense. |
| D18 | The vector index is IVF-PQ with refinement, with `nprobes=400` and `refine_factor=50`, and its recall is measured after every rebuild. |
| D20 | The tokenizer is byte-level BPE with 32,768 tokens, fitted on calls up to 2010, with digits split so that years cannot be dated by how they split. |
| D21 | Our encoder matches bge-small in size and trains on 1.2B tokens at every cutoff. |
| D22 | Contrastive fine-tuning caps questions at 256 tokens and answers at 512. |
| D23 | The contrastive batch, which is the number of negatives, is the same on every card, through gradient caching. |
| D26 | The forecasting protocol: a ridge regression refit each January on calls whose target was known, 32 principal components of the call vector, the penalty chosen by time-ordered cross-validation, and the next-quarter return as the first target. |
| D27 | Two forecasting targets that vary within a company, abnormal volatility after the call and the earnings surprise, under the D26 protocol. |
| D28 | The next-day return test on the full linked universe, with entry at the first close at least an hour after the call starts. |
| D29 | The chained family of vintages, and a from-scratch 2024 vintage at twice the token budget. |
| D30 | The nearest-neighbor forecast: the mean outcome of the most similar earlier calls of other firms. |
| D31 | The replay control: chained vintage Y continued on year Y for as many tokens as segment Y+1 reads. |
| D32 | The retrieval-augmented test, with ChronoGPT-Instruct of the cutoff year as the reader. |
| D33 | The 2024 look-ahead premium with the contrastive stage at three seeds, pretraining fixed. |
| D34 | The split of the look-ahead premium by how closely the firm's later calls restate each answer. The same decision covers tooling for pretraining-seed replicates of the chain and a fine-tuned bge-small reference, which the paper does not use. |

## Known limitations cited in the code and in the data dictionary

| | Limitation |
|---|---|
| L4 | The 0.094% gap between server-side and stored character counts is attributed to stripped trailing whitespace, which is plausible but untested. |
| L8 | Minor audit findings left as they are, among them that `permno` is arbitrary for the 21 Capital IQ companies that map to more than one. |
| L6 | The identity diagnostic rests on one draw with no interval. |
| L9 | A shared-cluster GPU task can land on hardware the job cannot use, so jobs check the device first. |
| L10 | The embeddings used for retrieval truncate about one chunk in five at 512 tokens. |
| L13 | bfloat16 is emulated on Turing cards, and `torch.cuda.is_bf16_supported()` does not say so. |
| L14 | Each vintage is specialized to its own period, which is the signal and also a trap. |
| L15 | The ChronoBERT comparison is not matched in capacity or in the amount of text read. |
| L16 | An early test of whether embeddings predict volatility levels is underpowered. |
| L17 | bfloat16 on Turing cards made domain-adaptive pretraining slow, and float16 was unavailable for it. |
