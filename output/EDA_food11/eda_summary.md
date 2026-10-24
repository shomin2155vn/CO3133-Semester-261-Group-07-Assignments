# Food-11 EDA summary (auto-generated)

### 1. Input size & data integrity
- 16,643 files indexed (train 9,866 / val 3,430 / test 3,347; 1133 MB); **16,643 decode fully, 0 unreadable**. Formats: JPEG; colour modes: RGB.
- Oddities: 0 non-RGB images, 0 extension/format mismatches, 1887 with EXIF orientation ≠ 1, 0 with short side < 64 px.
- Sizes are heterogeneous: 825 distinct (w, h) pairs; the most common is 512×512 (57.9 %). Longer side: median 512 px (min 270, max 9542); shorter side: median 512 px (min 207). Aspect w/h: median 1.00 (range 0.56–2.61); 58 % square, 30 % landscape, 12 % portrait.
- Resize to 512: 2.1 % up-sampled, 92.2 % unchanged longer side, 5.6 % down-sampled; 41.8 % need padding (mean 11.3 % of the canvas, max 62 %).
- Duplicates: exact cross-split shared hashes {'train-val': 51, 'train-test': 249, 'val-test': 311}; shared dHashes {'train-val': 58, 'train-test': 254, 'val-test': 314}; within-split exact duplicates {'train': 5, 'val': 3, 'test': 4}. **560 test image(s) (16.73 %) are byte-identical to a train/val image; 568 share a dHash with one** → possible optimistic test results; official splits are kept (no dedup). Label conflicts: 0 identical-byte groups, 0 identical-dHash groups.
### 2. Class distribution (whole dataset)
- N = 16,643 images in 11 classes; largest **Dessert** (2,500), smallest **Rice** (472). Split ratio train/val/test = 59.3 / 20.6 / 20.1 %.
- Val/test follow train: TV distance val 0.0284, test 0.0143; χ² (class × split) p = 0.806, Cramér V = 0.0208. Share of a class in train ranges 56.8–60.5 %.
### 3. Imbalance
- Label imbalance (max/min): train 5.36, val 5.21, test 5.21; normalised entropy (train) 0.956; effective classes 9.89 of 11; Gini 0.253.
- Trivial baselines (always “Dessert”): accuracy val 0.146 / test 0.149; macro-F1 val 0.023 / test 0.024; uniform chance 0.091.
- Input size vs class (Kruskal–Wallis effect sizes): long_side: ε²=0.184 (large); short_side: ε²=0.045 (small); aspect: ε²=0.046 (small); kb: ε²=0.166 (large); bpp: ε²=0.161 (large). Longer-side bucket × class Cramér V = 0.428. Mean padding per class ranges 7.9 % (Fried food) – 21.8 % (Vegetable-Fruit); median file size per class 41–133 KB.
- Pixel-free shortcut check: a random forest on size/aspect/file-size metadata only reaches accuracy 0.251 / macro-F1 0.193 on val (majority 0.146, chance 0.091) → only weak class signal; shortcut risk is limited.
### 4. Pixel intensity (train)
- **Train RGB mean = (0.5586, 0.4520, 0.3434)**, **std = (0.2722, 0.2755, 0.2821)** on [0,1] = mean (142.4, 115.3, 87.6), std (69.4, 70.3, 71.9) on [0,255]; 2,283,237,376 real-content pixels; padding colour (142, 115, 88).
- Difference to ImageNet constants: Δmean [0.074, -0.004, -0.063], Δstd [0.043, 0.052, 0.057]. Pixels at exactly 0 / 255: [0.48, 0.68, 2.67] % / [1.42, 1.05, 0.82] %.
- Per-image mean luminance 119.8 ± 29.0; contrast (luminance std) 58.8. Train flags: 32 dark, 41 bright, 27 low-contrast images.
- Class colour differences: brightest **Dairy product** (139), darkest **Meat** (111); warmest (R−B) Fried food, coolest Dairy product; class-mean spread per channel [0.0734, 0.1171, 0.1332].
- Val/test vs train (descriptive): max |Δmean| 0.0070, max |Δstd| 0.0034, largest KS D 0.031 → colour statistics are very similar across splits.
