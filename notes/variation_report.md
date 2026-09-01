# Capture variation report

Produced by `scripts/audit_variation.py` over 150 images.

## Brightness (grayscale mean per image, 0-255)
- mean 126.3, std 21.8, min 82.8, max 169.9
- histogram (8 bins over 0-255): 0 0 12 61 72 5 0 0

## Background diversity (HSV hist of non-box pixels, correlation-grouped)
- 10 background group(s) at correlation threshold 0.55
- group sizes: [np.int64(38), np.int64(32), np.int64(25), np.int64(14), np.int64(13), np.int64(8), np.int64(8), np.int64(6), np.int64(5), np.int64(1)]

## Box areas (fraction of frame) — distance-variation proxy
- min 1.69%, median 8.11%, max 33.95%
- max/min span: 20.1x

## Composition
- orientation mix: 147 portrait / 3 landscape
- boxes per image: mean 1.39, min 1, max 2
- instances: earphone_case=95, charger_brick=113

## Decision inputs (PLAN Phase 2c table)
- brightness std: 21.8 (wide if >~25)
- background groups: 10 (threshold in table: >=3)
- box-area span: 20.1x (threshold in table: >=10x = order of magnitude)
