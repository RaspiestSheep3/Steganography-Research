# Review of Composite Steganography Project

This document reviews the analysis paper, the Python implementation in `TestingScriptFuncVersion.py` and `GraphMakerNew.py`, and the seven graphs in `Graphs/`. The Helpers package (`HelperFunctions.py`, `EmbeddingAlgorithms.py`, `SteganalysisMethods.py`) was **not** present in the working directory, so anything that depends on the internal behaviour of `StandardLSB`, `LSBMatching`, `PixelPairMatching`, `ChiSquareAttack`, `ZhangLSBMatching` and `PSNR` is reasoned about from the paper rather than verified line-by-line. Items flagged as "verify in helpers" should be checked against those files.

---

## 1. Bugs and code-level issues

### 1.1 First block is used both for embedding **and** as the index block - SOLVED
`TestingScriptFuncVersion.py:48-49` saves `blocks[0][0]` as the index block:

```python
if(indexBlockMethod):
    indexBlock = deepcopy(blocks[0][0])
```

But `blocks[0][0]` is **also** the first block visited by the embedding loop (when `blockCounter == 0`), so the first secret section is embedded into it via the chosen pure method. At the end of `CompositeMethod`, line 141 overwrites that block:

```python
blocks[0][0] = indexBlock
```

So the embedding produced for the first secret section is silently discarded, and the recipient cannot recover that section. Effects:
- One block (1/16 at block-size 64) is dropped from the actual capacity.
- `methodsUsed[0]` reports the method picked for the first section, but no real data lives in `blocks[0][0]` other than the index payload.
- Slightly inflates the apparent security of the stego because one block actually carries no secret.

**Fix:** start the embedding loop at `blockCounter = 1`, slice `secretSplit` accordingly, and skip block (0,0) entirely. The index block selection (which block to reserve) should also be a parameter, since `(0,0)` is a corner with low texture and is a poor index location for many images.

### 1.2 Block-marking branch corrupts pixel values (bug, but only on the unused path) - SOLVED
`TestingScriptFuncVersion.py:148-149` (the `indexBlockMethod=False` branch):

```python
block[0][0] = (block[0][0] // 2) + int(binaryEmbedForMethodsUsed[counter])
block[0][1] = (block[0][1] // 2) + int(binaryEmbedForMethodsUsed[counter + 1])
```

This is **not** LSB replacement. For pixel value 200 (`11001000`):
- `200 // 2 == 100` (`01100100`)
- `+ 1 == 101` (`01100101`)

The pixel has been **halved**, not had its LSB rewritten. Correct LSB replacement is either:
- `(block[0][0] // 2) * 2 + int(bit)`, or
- `(block[0][0] & ~1) | int(bit)`.

This branch is currently disabled in the graph runs (`indexBlockMethod` defaults to `True`), so it is not affecting the current results, but it is a real bug that would destroy any stego produced via the block-marking variant.

### 1.3 Deviation coefficients are calibrated on *full images* but applied per *block* - EXPERIMENTING
The paper derives:
- `c = -1/580.76` from cover Chi² ≈ 1312 vs LSB-Matching stego Chi² ≈ 731
- `z = 1/10856.838` (see §2 below) from cover Zhang ≈ 6584 vs LSB-Matching stego Zhang ≈ 17441

These averages are the values you measure on the *whole* 256×256 image. Inside `CompositeMethod`, `ChiSquareAttack` and `ZhangLSBMatching` are run on individual 64×64 blocks (line 74-76), so `Chi2_stego - Chi2_cover` and `Zhang_stego - Zhang_cover` are roughly **1/16 of the magnitude** the coefficients were tuned against. The result is that `c·ΔChi²` and `z·ΔZhang` are ~16× too small relative to the PSNR term, so the metric is essentially `40/PSNR` with a tiny tie-break from the steganalysis terms. This is a major reason the metric heavily favours PPM (which has the highest PSNR of the three methods).

**Fix:** re-derive `c` and `z` from block-level statistics (run all three pure methods over a sample of 64×64 blocks and recompute the means), or scale the differences by the ratio of expected block-vs-image magnitudes.

### 1.4 PSNR can be `inf` and silently zeroes the PSNR term - DO NOT THINK ISSUE -> PSNR = INF MEANS NO CHANGES -> PERFECTLY SECURE
On lines 92, 95, 98 the metric contains `DEVIATION_COEFFICENTS["PSNR"] / PSNR(consideredBlock, stegoChanges[...])`. If a method makes zero changes (small section, all bits already match) `PSNR` returns `float('inf')` — `40/inf = 0`, which then **artificially makes that method look extremely secure** (tiny α). This can cause spurious method selection at low embed rates. A division-by-zero is also possible if the two arrays end up identical-but-not-detected.

**Fix:** clamp/handle `inf` explicitly, e.g. `psnr = min(PSNR(...), 100.0)` before using it.

### 1.5 Failure-handling never tracks the unembedded bytes
If `bestMapping >= acceptableMappingThreshold`, the section is dropped (line 110-111) — the block stays as cover, but `secretSplit[i]` is never re-assigned to a later block. Across all of the current graph runs the failure count is 0 (see §3.1 below) so this doesn't bite right now, but it is a correctness gap if you ever raise the threshold or lower it: the stego silently loses data instead of redistributing.

### 1.6 Non-random secret biases Chi² behaviour
`GraphMakerNew.py:43` reads `Lipsum.txt` as the secret. ASCII text has heavily biased bit values (lowercase letters cluster in `0110xxxx`/`0111xxxx`), so even at 100% embed rate the PoV pairs are not balanced. This is the most likely reason the Chi² curves for LSB and LSB-Matching **rebound after ~70%** in `Chi2 VS Embedding Rate Existing Methods 1000.png`, rather than approaching zero as theory predicts. Use cryptographically random bytes (`os.urandom`) as the secret if the goal is to characterise the embedding methods themselves; reserve text-style payloads for a separate "realistic payload" experiment.

### 1.7 Stale comment in `GraphMakerNew.py:199` - SOLVED
```python
#!THIS IS CURRENTLY SET FOR 32 * 32 - REMEBMER TO CHANGE FOR 64 * 64
```
The maths in the function does scale correctly with `blockSize` — `bytesPerSection = ceil(secretBytesAmount / ((imageSize[0] / blockSize[0]) ** 2))` — but the comment is misleading. Either delete it or move it to the call-site if there really is a manual switch you need to flip.

### 1.8 `TestingScriptFuncVersion.py` lives in `Helpers/` according to the import - NOT AN ISSUE
`GraphMakerNew.py:4` does `from Helpers.TestingScriptFuncVersion import CompositeMethod`, but the file you sent is `TestingScriptFuncVersion.py` at the project root. Make sure the file you actually run is the same one you're editing — drift between the two would silently invalidate any of the fixes below.

---

## 2. Sign of Zhang's coefficient - SOLVED, ABS()

**Short answer: the coefficient should be positive (`+1/10856.838`). Your code is correct (`TestingScriptFuncVersion.py:18`). The paper has a typo.**

Working through the paper's own derivation:
- Cover average `D_c ≈ 6583.76`
- LSB-Matching stego average `D_s ≈ 17440.598`
- The paper picks 17440.598 because it is the smallest of the three "increase" cases — the most conservative (least-punitive) calibration.
- `z · (D_s − D_c) = 1` ⇒ `z · (17440.598 − 6583.76) = 1` ⇒ `z · 10856.838 = 1` ⇒ **z = +1/10856.838**.

The paper instead writes "z = −1/10856.838", which contradicts its own equation. Your code uses the positive value, which is the value that actually maps "bigger Zhang increase ⇒ bigger contribution to α ⇒ less secure".

**However, this exposes a conceptual problem with the metric, not just a sign question.** The Zhang term punishes *increases* in Zhang's score and *rewards* decreases. Pixel-Pair Matching's Zhang's score *decreases* with embed rate (your own paper notes `D_s < D_c` for PPM, the Pixel Pair Matching curve in `Zhang VS Embedding Rate Existing Methods 1000.png` confirms it). With `z > 0`, every PPM block earns a **negative** contribution to α, i.e. the metric *credits* PPM for being detectable in the opposite direction.

**Recommended fix.** Use the magnitude of the deviation rather than the signed deviation:

```python
"PPM" : (
    abs(ChiSquareAttack(stegoChanges["PPM"]) - coverMappings["Chi Square Attack"]) * abs(DEVIATION_COEFFICENTS["Chi Square Attack"])
  + abs(ZhangLSBMatching(stegoChanges["PPM"]) - coverMappings["Zhang"]) * abs(DEVIATION_COEFFICENTS["Zhang"])
  + DEVIATION_COEFFICENTS["PSNR"] / PSNR(consideredBlock, stegoChanges["PPM"])
)
```

That way deviation in either direction (LSB-style upswing or PPM-style downswing) is treated as evidence of detectability and contributes positively to α. With this change you would expect PPM to stop dominating block selection at high embed rates and the curves in §3 to behave more like the pure methods.

---

## 3. Why does Setting **A** (threshold = 3, block size = 64) flatline?

### 3.1 What the graphs actually show

Looking at `Block Frequency VS Embedding Rate Composite Method A 1000.png`:
- Failures: ≈ 0 across the whole range (so the threshold is never binding for setting A)
- LSB Replacement: ≈ 3.5 → ≈ 0.5
- LSB Matching: ≈ 5 → ≈ 2
- **PPM: 6.6 → ≈ 13.3, saturating around 25-30% embed rate**

Total = 16 blocks (4 × 4 grid for block size 64), as expected. The plateau on PSNR/Chi²/Zhang starts at exactly the embed rate where PPM block selection saturates.

### 3.2 No, it is not normal — and there are at least three compounding causes

**(a) The metric is biased toward PPM.** As above, `z > 0` combined with `Zhang_stego_PPM < Zhang_cover` gives PPM a free negative contribution to α. PPM also has the highest PSNR of the three methods, so the dominant `40/PSNR` term also favours it. Once embed rate is high enough that LSB and Matching reach a "noticeable" deviation, PPM's α stays the lowest by a wide margin and is selected for almost every block.

**(b) PPM has half the bit-capacity of LSB / LSB-Matching.** PPM encodes one secret bit per pair of pixels (paper §3.1). For a 64×64 block:
- LSB / LSB-Matching capacity = 4096 bits = 512 bytes
- PPM capacity = 2048 bits = 256 bytes

`bytesPerSection = ceil(8192 · r/100 / 16) = ceil(5.12 · r)` bytes, so `bytesPerSection` exceeds PPM's 256-byte block capacity once **r > 50%**. Above that point PPM cannot embed the full section it has been given; what happens beyond capacity depends on how `PixelPairMatching` was implemented (truncation, wraparound, exception swallowed?). **You should check this directly** — if it silently truncates, then for r > 50% every PPM block embeds at most 256 bytes regardless of how much was requested, which is an obvious cause of saturation.

**(c) The plateau is not at 50% though — it starts at ~25%.** That suggests something more aggressive than PPM's nominal 1-bit-per-2-pixels capacity is being hit. Two candidates worth checking in `EmbeddingAlgorithms.py`:
- Does `PixelPairMatching` skip pixel pairs near the boundary of a block, or pairs where `(x, y)` are equal modulo something? Many PPM implementations only embed when a stego solution exists in the neighbourhood Ø(x,y), which can drop ~half the pairs in flat blocks.
- Does `PixelPairMatching` reserve part of the block for the index when `indexBlockMethod=True` is passed in? The third positional argument is being forwarded (line 85). If it reserves half the block, capacity drops to 128 bytes per block, which puts the saturation point at exactly **r ≈ 25%** — matching the observed flatline.

The third bullet is my best guess given the data. Confirming it requires reading `PixelPairMatching` in your `Helpers/EmbeddingAlgorithms.py`.

### 3.3 How to fix it

In rough order of payoff:

1. **Re-calibrate `c` and `z` against block-level statistics** (see §1.3). This alone should rebalance the three terms so PPM is not chosen by default.
2. **Use `abs(diff)` in the metric for the Zhang term** (see §2). Stops PPM gaming the metric by going in the other direction.
3. **Verify `PixelPairMatching`'s actual capacity for a 64×64 block** by calling it with a known-length payload and inspecting how much was embedded. If capacity is < 256 bytes per block, document it; ideally have the embedding method *return* the number of bits actually embedded, and have `CompositeMethod` redistribute leftover bytes to other blocks rather than silently dropping them.
4. **Cap the input section size to the chosen method's capacity** before the comparison, so all three candidate stegos are evaluated on the same payload size.
5. **Fix the (0,0) overlap (§1.1)** so the first block isn't double-purposed.
6. **Move to a random-bytes secret (§1.6)** for the characterisation graphs.
7. **Clamp infinite PSNR (§1.4)**.

Once 1, 2 and 3 are done, re-run setting A and the curves should track the pure methods rather than plateauing at ~25%.

---

## 4. Other paper-vs-code notes

- The paper's metric formula on page 5 reads `c·ΔChi² + z·ΔZhang + 40/PSNR = α`. The code matches this structure exactly (lines 90-99) — good.
- Paper refers to `α_T` as a threshold *above which* methods are insecure. Code does `if bestMapping < acceptableMappingThreshold` (line 106), which is consistent (lower α = more secure).
- Existing-method graphs (`*Existing Methods*.png`) match the paper's narrative well: Chi² drops then rebounds for LSB and LSB-Matching after ~70% (consistent with the secret-bias hypothesis in §1.6); Zhang increases for LSB-style methods and decreases for PPM, supporting the paper's discussion on PPM's anomalous Zhang behaviour.
- The Chi²/Zhang/PSNR coefficient discussion in §4 of the paper would be more compelling if it explicitly stated that the calibration was done at the *image* level. As §1.3 above notes, that mismatch is currently load-bearing.

---

## 5. Quick checklist

- [x] Index block conflicts with the first embedding block (§1.1)
- [x] Block-marking method has a wrong-LSB-replacement formula (§1.2)
- [x] Coefficients are calibrated on full images but applied per block (§1.3)
- [x] PSNR can be infinite and silently zeroes its term (§1.4)
- [x] Zhang coefficient sign — paper has typo, code is correct (positive) (§2)
- [x] Metric mathematically rewards PPM for going the other way on Zhang (§2)
- [x] PPM capacity at 64×64 likely matches the 25%-flatline (§3.2 (c)) — needs confirmation in helpers
- [x] Non-random secret biases Chi² high-rate behaviour (§1.6)
