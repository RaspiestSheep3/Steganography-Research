# Bug Report — Composite Steganography Research Scripts

**Reviewed:** 2026-05-02  
**Scope:** All `.py` files under `Adaptive Embedding Idea/` and `XOR Idea/`

---

## Summary

| # | File | Line(s) | Severity | Issue |
|---|------|---------|----------|-------|
| 1 | `Helpers/TestingScriptFuncVersion.py` | 137 | **Critical** | Off-by-factor-8 slice corrupts index block encoding | - SOLVED

| 2 | `HeatmapMaker.py` | 85–88, 320–322 | **Critical** | Multiline expression truncated — `threshold` only receives Chi² term | - SOLVED

| 3 | `LSBMatchingEmbedder.py` | 22–26 | **High** | `usedPoints` never appended — sampling with replacement | - SOLVED


| 4 | `TestingScript.py` | 44 | Low | `str.replace()` result discarded — newlines not removed | - SOLVED


| 5 | `Helpers/EmbeddingAlgorithms.py` | 145, 108–117, 172–173 | Low (dead code) | `MatrixEncoding` has 3 crash bugs if ever called |

---

## Bug 1 — Wrong Bit Slicing in Index Block Encoding

**File:** `Helpers/TestingScriptFuncVersion.py:137`  
**Severity:** Critical

### Code

```python
# BUGGY
for i in range(ceil(len(binaryEmbedForMethodsUsed) / 8)):
    binaryEmbedForMethodsUsedAsciiRepresentation += chr(int("".join(binaryEmbedForMethodsUsed[i:i+8]),2))

# FIXED
for i in range(ceil(len(binaryEmbedForMethodsUsed) / 8)):
    binaryEmbedForMethodsUsedAsciiRepresentation += chr(int("".join(binaryEmbedForMethodsUsed[i*8:i*8+8]),2))
```

### Explanation

`binaryEmbedForMethodsUsed` is a flat list of bits e.g. `[0,0, 0,1, 1,0, ...]` where every 2-bit pair encodes one block's embedding method. To convert it to characters for LSB embedding, each group of 8 consecutive bits should be read as one byte.

The bug: loop variable `i` is used as both the **byte index** and the **start of the bit slice**, but it should be `i*8`. The actual slices produced are:

| Iteration `i` | Buggy slice | Correct slice |
|---------------|-------------|---------------|
| 0 | `[0:8]` ✓ | `[0:8]` |
| 1 | `[1:9]` ✗ | `[8:16]` |
| 2 | `[2:10]` ✗ | `[16:24]` |
| 3 | `[3:11]` ✗ | `[24:32]` |

All bytes after the first overlap with each other and are computed from wrong bits.

### How It Shows Up in Results

The **index block** is the mechanism the receiver uses to know which embedding method (LSB, Matching, PPM, or Failure) was used in each image block. A corrupt index block means:

- The receiver reads wrong method codes for all blocks beyond the first.
- It attempts to decode data using the wrong algorithm — extraction produces garbage.
- The steganographic scheme **appears to work** during embedding and steganalysis testing (because those don't test decoding), but the system is **non-functional as a communication channel**.
- If the student runs extraction tests as a future step, all decoded secrets will be garbled.

---

## Bug 2 — Multiline Expression Silently Truncated

**File:** `HeatmapMaker.py:85–88` and `HeatmapMaker.py:320–322`  
**Severity:** Critical  
**Affects:** `GenerateBlockSizeHeatmapData()` and `GenerateThresholdHeatmapData()`

### Code

```python
# BUGGY — Python sees 3 separate statements, not one expression
threshold = (ChiSquareAttack(stegoBlock) - coverMappings["Chi Square Attack"]) * DEVIATION_COEFFICENTS["Chi Square Attack"] 
+ (ZhangLSBMatching(stegoBlock) - coverMappings["Zhang"]) * DEVIATION_COEFFICENTS["Zhang"] 
+ DEVIATION_COEFFICENTS["PSNR"] / PSNR(consideredBlock, stegoBlock),

# FIXED — wrap in parentheses for implicit line continuation
threshold = (
    (ChiSquareAttack(stegoBlock) - coverMappings["Chi Square Attack"]) * DEVIATION_COEFFICENTS["Chi Square Attack"]
    + (ZhangLSBMatching(stegoBlock) - coverMappings["Zhang"]) * DEVIATION_COEFFICENTS["Zhang"]
    + DEVIATION_COEFFICENTS["PSNR"] / PSNR(consideredBlock, stegoBlock)
)
```

### Explanation

Python does not allow implicit line continuation outside of brackets/parentheses. Without a trailing `\` or enclosing `()`, each physical line is a separate statement. So Python parses this as:

1. `threshold = <chi2_term>` — assignment of only the Chi² component (also a 1-tuple due to trailing comma)
2. `+ <zhang_term>` — unary plus applied to Zhang result; evaluated and **silently discarded**
3. `+ <psnr_term>,` — evaluated and **silently discarded**

The composite security metric `α` is designed as a weighted combination of three signals:
- **Chi² attack delta** — measures LSB replacement detectability
- **Zhang delta** — measures LSB matching detectability  
- **PSNR contribution** — measures visual distortion

With only the Chi² term being stored, `threshold` is:
- Not normalised correctly (missing Zhang and PSNR contributions)
- Also stored as a **1-tuple** `(value,)` not a float, which would cause a `TypeError` when added to `totalThresholdSum` on the next line

The script likely crashes at runtime, or produces entirely wrong heatmap data if it somehow proceeds.

### How It Shows Up in Results

The `α` (security) heatmap data stored in the SQLite database is meaningless — it reflects only Chi² delta, not the composite metric. This affects:

- **Block Size heatmaps** (`Block Size VS Embedding Rate Composite Alpha Heatmap 1000 *.png`) — security scores wrong
- **Threshold heatmaps** (`Block Size VS Threshold Composite Alpha Heatmap 1000 *.png`) — security scores wrong
- Any conclusions drawn about optimal block size or threshold from these heatmaps are unreliable

If the `TypeError` from the 1-tuple addition causes a crash before `conn.commit()`, the database tables will be empty or partial.

---

## Bug 3 — `usedPoints` Never Populated (Sampling With Replacement)

**File:** `LSBMatchingEmbedder.py:22–26`  
**Severity:** High

### Code

```python
# BUGGY — usedPoints.append(point) is missing
usedPoints = []
for bit in embed:
    point = (None, None)
    while(point == (None, None) or point in usedPoints):
        point = (random.randint(0,255), random.randint(0,255))
    # usedPoints.append(point)  <-- THIS LINE IS MISSING
    
    value = cover.getpixel(point)
    ...

# FIXED — add the append after selecting the point
usedPoints = []
for bit in embed:
    point = (None, None)
    while(point == (None, None) or point in usedPoints):
        point = (random.randint(0,255), random.randint(0,255))
    usedPoints.append(point)
    
    value = cover.getpixel(point)
    ...
```

### Explanation

`usedPoints` is intended to track already-used pixels so each pixel is embedded into at most once. Because the append is missing, `usedPoints` stays empty forever. The `while` loop exits as soon as any non-`(None, None)` point is generated (first iteration), and the same pixel coordinates can be selected again on subsequent bits. The last bit written to any pixel wins — all earlier writes to that pixel are overwritten.

### How It Shows Up in Results

This affects the **LSB matching stego images** used as ground-truth stegos for calibrating the Zhang steganalysis detector (`LSBMatchingAnalyser.py`). Specifically:

- Effective embed rate is lower than intended — many bits are written to pixels that already hold a different embedded bit
- The actual number of pixels modified is fewer than `EMBED_AMOUNT * 8` (32,768 pixels for 4096 bytes)
- `ZhangLSBMatching` scores on these stegos will show **less smoothing** than a correctly-embedded image at the same embed rate
- The derived coefficient for Zhang in `TestingScriptFuncVersion.py` (`DEVIATION_COEFFICENTS["Zhang"] = 1/10856.838`) was likely calibrated against these flawed stegos
- If the bug is fixed, recalibration is needed

---

## Bug 4 — `str.replace()` Result Discarded

**File:** `TestingScript.py:44`  
**Severity:** Low

### Code

```python
# BUGGY — str.replace() is immutable, returns new string
secret.replace("\n","")

# FIXED
secret = secret.replace("\n","")
```

### Explanation

Strings in Python are immutable. `str.replace()` returns a new string with substitutions applied; it does not modify the original in-place. The result is not assigned, so `secret` still contains newline characters.

### How It Shows Up in Results

`TestingScript.py` is an older standalone script (uses hardcoded paths like `"Flapjack.png"`), not part of the main analysis pipeline. Impact is limited to this script:

- Newline characters (`\n`, ASCII 10) get embedded as part of the secret
- Slightly alters embed rate and the specific bit patterns embedded
- Negligible effect on Chi²/Zhang/PSNR measurements since newlines are just another byte value

---

## Bug 5 — Dead Code Bugs in `MatrixEncoding`

**File:** `Helpers/EmbeddingAlgorithms.py`  
**Severity:** Low (function is never called or imported anywhere)

The `MatrixEncoding` function has a comment noting it is not used: `#*Note - not using Matrix Encoding because I do not fully understand it`. It is not imported in any other script. However, if it is ever activated, it will crash:

### Bug 5a — Formatting a list instead of an int (line 145)

```python
pi = [blockFlattened[i]]          # pi is a list: [int]
piBits = [*format(pi, "08b")]     # TypeError: unsupported format character

# Fix:
pi = blockFlattened[i]            # pi should be the int directly
piBits = [*format(pi, "08b")]
```

### Bug 5b — Float values passed to `range()` (lines 108–117)

```python
N3 = L/3          # float division → N3 is a float
...
Is = [x for x in range(N1)]      # TypeError: 'float' object cannot be interpreted as integer

# Fix: use integer division
N3 = L//3
N1 = int((2*L/3) - H * W)
N2 = int(L - H * W)
```

### Bug 5c — Hardcoded block size in deflattening (lines 172–173)

```python
# BUG: hardcodes 64 rows of 64 pixels — only correct for 64×64 blocks
newBlock = []
for i in range(64):
    newBlock.append(newBlockFlattened[64*i:64*i+64])

# Fix: use actual block dimensions
H = len(block)
W = len(block[0])
newBlock = []
for i in range(H):
    newBlock.append(newBlockFlattened[W*i:W*i+W])
```

### How It Shows Up in Results

No current impact — function is dead code. If reactivated without fixing: immediate `TypeError` crash before any embedding occurs.

---

## Fix Priority

| Priority | Action |
|----------|--------|
| 1 | Fix Bug 2 (`HeatmapMaker.py`) — re-run heatmap data generation, re-check stored DB values |
| 2 | Fix Bug 1 (`TestingScriptFuncVersion.py`) — re-validate index block method results |
| 3 | Fix Bug 3 (`LSBMatchingEmbedder.py`) — regenerate LSB matching stego database, recalibrate Zhang coefficient |
| 4 | Fix Bug 4 (`TestingScript.py`) — low impact, simple fix |
| 5 | Fix Bug 5 (`EmbeddingAlgorithms.py`) — only if `MatrixEncoding` is to be used |
