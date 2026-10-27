# BEGAD feature engineering and statistical discovery: Claude Code entry point

Ngày cập nhật: 27/10/2026. Đọc toàn bộ bảy file trong thư mục này trước khi triển khai.

Mục tiêu: giữ **160 ứng viên gốc (80 micro + 80 macro)** và yêu cầu Claude tự
đọc dataset, đề xuất thêm **100 features**. Tổng mục tiêu là **260 định nghĩa**;
tỷ lệ micro/macro của 100 features mới tùy dữ liệu thực có. Sau đó kiểm tra và xuất:

1. Các macro features được chọn, kèm task/phạm vi và bằng chứng thống kê.
2. **Top 20 cặp task–ROI**, mỗi cặp kèm các micro features được chọn.
3. Bảng xếp hạng các task, danh sách dự phòng theo từng cặp chẩn đoán và audit các feature không dùng được.

Đây là yêu cầu triển khai feature engineering và statistical discovery, **không train classifier, normative neural network, MIL hay benchmark ML**. Chưa có kết quả chọn feature thật trong bộ hướng dẫn này.

## Prompt to paste into Claude Code

```text
Work in the BEGAD_Norm repository. Read this entire instruction pack,
starting with 00_START_HERE.md, then execute its implementation and analysis.
Inspect AGENTS.md and the current repository before modifying code. Preserve
unrelated work and raw data. Preserve the 160 base definitions, then independently
propose 100 additional dataset-grounded features under
06_DATASET_DRIVEN_100_FEATURES.md. Inspect the actual sources first; do not ask
the researcher to supply the extra feature names. Use D001–D100, each explicitly
micro or macro. Do not pad the list with renamed features, ROI/task instances,
rank-equivalent transforms or unmeasured signals. Document a genuine shortfall.

Freeze the additional definitions and thresholds before inspecting diagnostic
p-values. Implement the combined registry (target 260), audited extraction,
subject-level statistical screening, and task–ROI ranking. Both feature origins
must enter the same relevant multiplicity families and final selection lists.
Do not train classification models.

Use CN as the report label for the repository's CO/HC control class, after
verifying the actual group mapping. Never infer diagnostic labels from row order.

Default exploratory selection: feasible AND p_CN_AD < 0.20 AND p_CN_MCI < 0.20.
Always test/report MCI_AD too. Also export an all-three-pairs < 0.20 subset and
pair-specific reserve lists. Do not silently replace this rule with OR or require
all three comparisons for the main list. Report FDR, effect sizes, uncertainty,
missingness and stability alongside the permissive screen.

For micro features, each hypothesis is one (dataset, task, ROI, feature, contrast),
with at most one scalar per independent subject. Rank up to 20 unique task–ROI
pairs, not 20 individual feature rows. Do not force-fill the list when fewer
than 20 qualify. Keep different cohorts and incompatible exports separate.

Run the implementation, tests, available-data extraction, statistics and reports.
If some sources are absent, mark dependent candidates unavailable and complete
the supported analyses. Report real blockers; never fabricate measurements,
p-values or selected regions. Produce the Vietnamese results memo and the
machine-readable outputs specified in 04_ROI_RANKING_AND_REPORTS.md.
```

## Read order and authority

| File | Responsibility |
|---|---|
| `01_FEATURE_CATALOG_160.md` | Exactly 80 micro and 80 macro definitions, dependencies and units |
| `02_DATA_AUDIT_AND_EXTRACTION.md` | Cohorts, semantic validation, event reconstruction, missingness and data schemas |
| `06_DATASET_DRIVEN_100_FEATURES.md` | Claude independently proposes 100 additional definitions from inspected data, audits novelty/support and freezes the expanded catalogue |
| `03_STATISTICAL_SCREENING.md` | Primary tests, thresholds, multiplicity, effect sizes, uncertainty and stability |
| `04_ROI_RANKING_AND_REPORTS.md` | ROI/task ranking, selection tables and final conclusions |
| `05_IMPLEMENTATION_AND_ACCEPTANCE.md` | Code organization, CLI, meaningful tests and completion checks |

The numeric protocol in file 03 is authoritative. The catalogue contains proposed
measurements, not a claim that every feature is a validated AD biomarker or that
all base or additional candidates can be extracted from the currently available
workbooks. File 06 runs after the source audit and before diagnostic screening,
despite its numerical filename. Its 100 definitions are to be authored by Claude
from the real data; they are not already supplied in this instruction pack.

## Scope and starting evidence

Repository last inspected at commit `652567263a7c9727b15f281c4e96f6bde9d97106`.
Recheck current HEAD; adapt paths if changed and record differences.

- Raw/task-level branch: 157 baseline participants, previously 100 CO / 41 MCI /
  16 AD. Existing 25 selected features are **not** the candidate universe.
- T5_clean.xlsx previously read directly: 6,217 rows, 64 participants, 32 CO /
  17 MCI / 15 AD, reading task 5, 159 ROI identifiers.
- T1235_clean.xlsx configuration reports 12,486 rows, 64 participants and tasks
  1/2/3/5. Recompute these counts from the actual workbook; do not treat config
  comments as verified data.
- Workbook tasks 1/2/3 are KD cards and task 5 is reading. Raw task numbering is
  different: KD 3/4/5 and reading 9 in the inspected snapshot. Verify through
  stimulus metadata before assigning canonical names.
- Raw157 and ROI64 must remain separate until person/visit/stimulus identities
  and feature definitions are demonstrably compatible. More rows are not more
  participants.

Primary ROI ranking uses the verified combined T1235 cohort if its sources are
usable. T5 is a separate reading consistency/sensitivity analysis until equivalence
is demonstrated; never count duplicate reading records as additional people or
as additional distinct top-20 regions. If T1235 cannot be read, complete T5 and
label the resulting ranking **reading-only**, not combined KD+reading.

## Deliverable naming

Use a new run directory, e.g. `outputs/feature_discovery/<run_id>/`, with immutable
input hashes and a frozen configuration. Do not overwrite earlier model results.
The final user-facing memo must distinguish:

- computationally available;
- technically reliable enough to test;
- exploratory selected under raw p < 0.20;
- supported after multiplicity correction;
- stable/unstable across subject resampling;
- unavailable or inconclusive.

Selection on this full dataset is exploratory. Any later predictive evaluation
must repeat label-dependent feature/ROI selection inside the training data or
use an independent confirmation cohort. Do not report classifier accuracy here.
