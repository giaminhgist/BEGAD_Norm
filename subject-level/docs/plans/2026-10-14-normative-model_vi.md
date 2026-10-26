# Mô hình chuẩn hóa học được cho BEGAD (CO/MCI/AD) — Kế hoạch triển khai

> **Cho người triển khai:** BẮT BUỘC: dùng entry `/minh-agent:code` để triển khai kế hoạch
> này theo từng task. Các bước dùng checkbox (`- [ ]`) để theo dõi.

**Mục tiêu:** Port pipeline learned normative-deviation của LNGModel (EXP-PROP-001,
`mlp_deepset`) sang BEGAD_Norm và đánh giá theo đúng protocol baseline hiện có
(3 bài toán × CV 5-fold theo participant × 20 seeds) để so sánh trực tiếp với
`outputs/baseline_balanced`.

**Kiến trúc:** Package mới `src/begad_norm/normative/`: ma trận per-participant
(11 task × 25 features) kèm mask → encoder 25→128→128 → ngân hàng chuẩn latent CO
làm mới mỗi epoch (μ/σ theo task trên encoding của CO trong fold train) → comparator học được
trên `[z, μ, z−μ, z·μ]` → deviation 64-d theo task → masked deepset pooling (mean‖max) →
embedding chủ thể 128-d → MLP head (sigmoid nhị phân / softmax 3-class). AdamW, early stopping
theo val AUC, checkpoint tốt nhất = val AUC cao nhất (epoch sớm nhất khi hòa), reseed mô hình
ở đầu mỗi fold.

**Tech Stack:** Python ≥ 3.12, PyTorch (chỉ CPU; dependency mới, đã được duyệt 2026-10-14),
numpy/pandas/scikit-learn/pyyaml (có sẵn), uv, pytest.

**Spec:** chính tài liệu này — §1 Spec thiết kế, §2 Các task.

---

## §1 Spec thiết kế

### 1.1 Quyết định đã duyệt (2026-10-14)

- Thêm `torch` (bản CPU) vào BEGAD_Norm bằng `uv add torch` — máy không có GPU (8 nhân CPU).
- Phạm vi: chỉ biến thể chính `mlp_deepset`. Các ablation (`mlp_mean`, `z_mean`, …) ngoài phạm vi.
- Stimulus = cả 11 task của protocol (so sánh được với 275 cột của baseline). Biến thể chỉ
  6 task kích thích để lại làm tùy chọn cấu hình, chưa chạy.

### 1.2 Protocol — phải khớp baseline chính xác

- Các bài toán (giống hệt `src/begad_norm/baseline/config.yaml`):
  - `co_vs_mci` — nhị phân, positive = MCI;
  - `co_vs_ad` — nhị phân, positive = AD;
  - `co_mci_ad` — 3-class (CO=0, MCI=1, AD=2): Sen/Precision/Spec/F1 là macro average,
    AUC là macro one-vs-rest, Acc là accuracy tổng.
- CV: theo participant, phân tầng theo group, 5 fold, tạo bởi CÙNG hàm với baseline —
  import `stratified_folds` từ `begad_norm.baseline.run_baselines`
  (hoán vị trong từng group bằng rng(seed), gán round-robin). Không viết lại.
- Seeds: 20 seeds của baseline
  `[42, 2026, 1, 0, 7, 123, 1234, 2024, 2025, 1337, 2, 3, 5, 10, 21, 777, 2020, 2021, 2022, 2023]`
  (khóa trong config.yaml).
- Một mẫu mỗi participant. Ngưỡng cố định 0.5 (nhị phân); argmax (3-class). Không chỉnh
  ngưỡng, không chỉnh siêu tham số.
- Metrics mỗi lần đánh giá: `acc, sen, precision, spec, auc, f1` — cùng định nghĩa với
  `metrics_binary` / `metrics_multiclass` của baseline.
- Schema detail CSV phải bằng `outputs/baseline_balanced/results/results_detail.csv`:
  `dataset, problem, model, seed, fold, n_train, n_test, acc, sen, precision, spec, auc, f1`.

### 1.3 Missingness và chuẩn hóa (port từ LNGModel `src/data/tabular.py`)

- Dòng (participant, task) có cả 25 feature đều NaN là task thiếu → bị mask ở mọi nơi
  (pooling, ngân hàng, loss). Ground truth cho test: cột `n_tasks_present` trong
  `processed_dataset/metadata.csv` (BEGCO-029 và BEGMCI-019 thiếu task 10).
- Ô NaN trong một dòng hợp lệ được điền bằng mean per-feature của fold train, nên đóng góp
  deviation bằng 0 sau chuẩn hóa.
- Chuẩn hóa: mean/std per-feature tính chỉ trên các dòng hợp lệ của các participant trong
  fold train (unsupervised, an toàn leakage); std chặn dưới EPS = 1e-6.
- Mask task được xác định từ các dòng toàn-NaN TRƯỚC mọi thao tác khác.

### 1.4 Mô hình (port từ LNGModel `src/models/proposal.py`, siêu tham số khóa)

| Thành phần | Đặc tả |
|---|---|
| encoder | Linear(25,128) → LayerNorm → GELU → Linear(128,128) |
| ngân hàng (bank) | buffer `bank_mu`, `bank_sigma` (11 × 128), làm mới MỖI epoch từ encoding của CO trong fold train; μ/σ có trọng số theo task trên các dòng CO hợp lệ; task không có CO → μ=0, σ=EPS |
| comparator | `[z, μ, z−μ, z·μ]` (512-d) → Linear 256 → ReLU → Linear 128 → ReLU → Linear 64 |
| pooling | deepset: masked mean ‖ masked max → 128-d |
| head | Linear(128,64) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(64, 1) sigmoid (nhị phân) / Linear(64, 3) softmax (3-class) |
| optimizer | AdamW, lr 1e-3, weight_decay 1e-4 |
| lịch huấn luyện | tối đa 150 epoch, patience 30 theo val AUC, batch size 16, dropout 0.3 |
| loss | nhị phân: BCELoss trên đầu ra sigmoid; 3-class: CrossEntropyLoss trên logit. Class weights = tần suất nghịch đảo theo fold train (khớp `class_weight: balanced` của baseline). |
| checkpoint | tốt nhất = val AUC cao nhất (epoch sớm nhất khi hòa); `best.pt`/`last.pt` gồm cả buffer bank; khôi phục phải tái lập chính xác dự đoán val |

### 1.5 Hành vi trainer (port từ LNGModel `src/training/trainer.py`)

- `seed_all(cfg.seed)` TRƯỚC khi dựng mô hình; reseed ở đầu mỗi fold để các fold độc lập
  với thứ tự chạy.
- `refresh_bank(train_ds)` gọi trước mỗi epoch (bank trễ một epoch; checkpoint tốt nhất chứa
  đúng bank mà metrics val của nó được tính).
- Metrics train/val cuối epoch tính ở chế độ `eval()` trên TOÀN BỘ subject của mỗi split.
- Early stopping theo val AUC (nhị phân: ROC-AUC của lớp positive; 3-class: macro OVR AUC).
  AUC val NaN → hủy run với lý do tường minh (không bao giờ âm thầm thay bằng 0.5).
- Artifacts mỗi run dir: `metrics.jsonl`, `best.pt`, `last.pt`, `predictions.csv`,
  `summary.json`, `run_info.json` (git commit, code hash, data hashes, env, danh sách
  train/val ids, status).

### 1.6 Đầu ra

```
outputs/normative/
├── runs/{problem}/{model}/seed{s}/fold{f}/   # artifacts từng run (như §1.5)
├── results/results_detail.csv                # một dòng mỗi evaluation (schema baseline)
├── results/summary.csv                       # qua aggregate_results.py của baseline
├── run_log.csv                               # trạng thái job, hashes, thời lượng
└── provenance.json                           # config sha, data shas, env, git
```

Summary tính bằng `begad_norm.baseline.aggregate_results` sẵn có
(`--expected-rows 300 --per-group 100`), không viết lại.

### 1.7 Ngân sách chạy

3 bài toán × 5 fold × 20 seeds = 300 lần huấn luyện. Chỉ CPU, ~112 subject train,
~7 batch/epoch, ≤150 epoch → ước tính ~20–60 phút wall với 8 workers
(torch intra-op threads = 1 mỗi worker).

## Ràng buộc toàn cục

- Không có GPU trên máy này → torch chỉ CPU. `uv add torch` đã được duyệt (quyết định user).
- Dùng lại nguyên văn cách tạo fold, danh sách seed, ngữ nghĩa metric của baseline — không
  sao chép kèm sửa đổi.
- Không chỉnh ngưỡng/siêu tham số; không có test set; mọi metric là metric validation.
- Determinism: cùng seed → cùng kết quả (cùng gán fold và khởi tạo mô hình).
- Không sửa `processed_dataset/`, code baseline, hay output baseline hiện có. Code mới chỉ
  nằm trong `src/begad_norm/normative/`, `tests/`, `docs/`.
- Markdown sinh ra phải có bản tiếng Anh + tiếng Việt.

## Trọng tâm review

1. **Rò rỉ subject val** vào bank, imputation, hoặc chuẩn hóa — mọi thống kê chỉ fit trên
   subject của fold train; bank chỉ trên CO của fold train.
2. **Dòng task toàn-NaN** phải bị mask, không bao giờ impute thành quan sát thật.
3. **Fold suy biến** (val chỉ một lớp) → AUC NaN phải hủy run với lý do rõ ràng.
4. **Tính đúng 3-class** — xác suất tổng bằng 1, dự đoán argmax, macro Sen/Spec/F1 và
   OVR macro AUC phải khớp định nghĩa baseline.
5. **Khôi phục checkpoint** phải tái lập chính xác dự đoán val (buffer bank nằm trong
   state dict).
6. **Lớp positive nhị phân** phải là positive trong config (MCI cho `co_vs_mci`, AD cho
   `co_vs_ad`) — kiểm tra dấu sen/spec đối chiếu confusion matrix.

---

## §2 Các task

### Task 1: Dependency + khung package

**Files:**
- Sửa: `pyproject.toml` (thêm `torch`)
- Tạo: `src/begad_norm/normative/__init__.py`, `src/begad_norm/normative/seeding.py`,
  `src/begad_norm/normative/config.yaml`

**Interfaces:**
- Sinh ra: `seeding.seed_all(seed)`, `seeding.code_hash()`, `seeding.determinism_env_info()`
  (seed python/numpy/torch; sha256 các file trong package normative; dict thông tin env).

- [ ] **Bước 1:** `uv add torch`; kiểm chứng bằng
  `uv run python -c "import torch; print(torch.__version__)"` (wheel CPU).
- [ ] **Bước 2:** Chạy lại test suite hiện có để chắc không vỡ gì: `uv run pytest -q`.
- [ ] **Bước 3:** Viết `tests/test_normative.py::test_seed_all_deterministic`:
  hai lần `seed_all(42)` → cùng trọng số torch.
- [ ] **Bước 4:** Cài `seeding.py`; chạy `uv run pytest tests/test_normative.py -v` → pass.
- [ ] **Bước 5:** Commit.

### Task 2: Tầng dữ liệu

**Files:**
- Tạo: `src/begad_norm/normative/data.py`
- Test: `tests/test_normative.py`

**Interfaces:**
- Dùng: `processed_dataset/features.csv`, `processed_dataset/metadata.csv`,
  `begad_norm.config.GROUP_LABELS`.
- Sinh ra:
  ```python
  def build_matrices(participants, features_csv, feature_cols, n_tasks=11)
      -> dict[pid, (X: (S, D) np.float32, mask: (S,) bool)]     # mask = dòng không toàn-NaN
  def feature_norm_stats(train_pids, matrices) -> (mu (D,), std (D,))   # nanmean/nanstd, std>=EPS
  class NormativeDataset:
      def __init__(self, participants, train_participants, features_csv, metadata_csv,
                   problem, n_tasks=11)
      # thuộc tính: X (P,S,D) float32 đã chuẩn hóa + điền, mask (P,S) float32,
      #             subjects (list[str]), labels (np.ndarray), positive (str|None)
      def co_indices(self) -> list[int]      # chỉ số các subject CO trong fold train
      def __getitem__(self, idx) -> ((X_i, mask_i), label, subject_id)
      def collate(self, items) -> ((B,S,D), (B,S))
  ```

- [ ] **Bước 1:** Viết test fail: shape `(157, 11, 25)`; `mask.sum(1)` bằng
  `n_tasks_present` trong metadata; không còn NaN trong `X`; các ô train hợp lệ sau chuẩn
  hóa có mean ≈ 0; stats không đổi khi thêm participant val vào lời gọi stats; labels
  `{0,1}` với MCI=1 cho `co_vs_mci`, `{0,1,2}` cho `co_mci_ad`; `co_indices()` chỉ chứa
  subject CO của train.
- [ ] **Bước 2:** Chạy test → FAIL.
- [ ] **Bước 3:** Cài `data.py` (pivot long→wide, mask, điền mean, chuẩn hóa bằng stats
  train, ánh xạ nhãn từ `group_label`).
- [ ] **Bước 4:** `uv run pytest tests/test_normative.py -v` → pass.
- [ ] **Bước 5:** Commit.

### Task 3: Mô hình

**Files:**
- Tạo: `src/begad_norm/normative/model.py`
- Test: `tests/test_normative.py`

**Interfaces:**
- Dùng: `NormativeDataset` (Task 2).
- Sinh ra:
  ```python
  class NormativeModel(nn.Module):
      def __init__(self, d_in=25, latent_dim=128, dev_dim=64, comparator="mlp",
                   pool="deepset", dropout=0.3, n_stim=11, n_classes=2)
      @torch.no_grad()
      def refresh_bank(self, train_ds: NormativeDataset) -> None   # μ/σ theo task chỉ trên CO
      def forward(self, x, y=None) -> (prob (B,1)|(B,C), {"emb": h})
  ```

- [ ] **Bước 1:** Viết test fail: forward nhị phân → `(B,1)` trong [0,1]; forward 3-class →
  `(B,3)` tổng mỗi hàng bằng 1; `refresh_bank` bằng mean/std có trọng số tính tay trên
  encoding latent của CO (allclose) và bỏ qua các hàng không phải CO; `bank_mu`/`bank_sigma`
  xuất hiện trong `state_dict()`.
- [ ] **Bước 2:** Chạy test → FAIL.
- [ ] **Bước 3:** Cài `model.py` port trực tiếp từ LNGModel `proposal.py` với các thay đổi
  §1.4 (head n_classes, masked pooling giữ nguyên).
- [ ] **Bước 4:** `uv run pytest tests/test_normative.py -v` → pass.
- [ ] **Bước 5:** Commit.

### Task 4: Trainer

**Files:**
- Tạo: `src/begad_norm/normative/trainer.py`, `src/begad_norm/normative/metrics.py`
- Test: `tests/test_normative.py`

**Interfaces:**
- Dùng: `NormativeDataset`, `NormativeModel`, `seeding`.
- Sinh ra:
  ```python
  @dataclass
  class NormativeConfig:   # problem/model/seed/fold/epochs/patience/lr/weight_decay/
                           # batch_size/dropout/n_classes/... + to_dict/from_dict/save/load/run_dir
  def compute_metrics(y_true, y_score, n_classes, positive=None) -> dict
      # nhị phân: metrics_binary của baseline (sen/spec/precision/f1 của positive, ROC AUC, acc)
      # 3-class: metrics_multiclass của baseline (macro sen/spec/precision/f1, OVR macro AUC, acc)
  def train_model(model_factory, train_ds, val_ds, cfg, device) -> dict
      # summary: best_epoch, stopping_reason, best_val_metrics, val ids, history
  ```

- [ ] **Bước 1:** Viết test fail trên dữ liệu tổng hợp nhỏ (~40 subject, 2 lớp):
  chạy lại cùng seed → cùng best AUC; khôi phục checkpoint tái lập dự đoán val; early
  stopping kích hoạt đúng patience; val chỉ một lớp → RuntimeError nêu lý do;
  `compute_metrics` khớp sklearn reference cho cả nhị phân và 3-class.
- [ ] **Bước 2:** Chạy test → FAIL.
- [ ] **Bước 3:** Cài `metrics.py` + `trainer.py` theo §1.5 (port trainer của LNGModel,
  thêm loss có trọng số lớp, hỗ trợ head 3-class).
- [ ] **Bước 4:** `uv run pytest tests/test_normative.py -v` → pass.
- [ ] **Bước 5:** Commit.

### Task 5: Runner + config

**Files:**
- Tạo: `src/begad_norm/normative/run_normative.py`
- Sửa: `src/begad_norm/normative/config.yaml` (khung từ Task 1)

**Interfaces:**
- Dùng: tất cả các phần trên + `stratified_folds` import từ
  `begad_norm.baseline.run_baselines`.
- Sinh ra (CLI): `uv run python -m begad_norm.normative.run_normative
  --config src/begad_norm/normative/config.yaml [--smoke] [--problems ...] [--seeds ...]
  [--folds ...] [--workers 8]`.
  Mỗi job `(problem, seed, fold)`: fold từ `stratified_folds(pid_group, seed, 5)`; dựng
  `NormativeDataset` train/val (không trùng participant, có assert); huấn luyện; ghi thêm
  một dòng vào `results_detail.csv` (schema baseline, flush từng dòng); cập nhật
  `run_log.csv`; ghi `provenance.json`. Resume-safe: bỏ qua job chỉ khi run dir của nó có
  `summary.json` trạng thái completed VÀ code/data hash khớp file hiện tại.

- [ ] **Bước 1:** Cài `run_normative.py` + hoàn thiện `config.yaml` (3 bài toán, 20 seeds,
  siêu tham số khóa, đường dẫn đầu ra như §1.6).
- [ ] **Bước 2:** Smoke: `--smoke` (co_vs_mci × seed 42 × fold 1, 5 epochs) chạy end-to-end,
  run dir đủ artifacts, có một dòng detail.
- [ ] **Bước 3:** Mini matrix: `--problems co_vs_mci --seeds 42 --workers 4` → 5 dòng detail
  (mỗi fold một dòng); kiểm tra tên cột bằng detail CSV của baseline; kiểm tra train/val
  rời nhau trong `run_info.json` từng run.
- [ ] **Bước 4:** Kiểm chứng chéo: `uv run python -m begad_norm.baseline.aggregate_results
  --detail <mini-detail> --summary <tmp> --expected-rows 5 --per-group 5` chạy được.
- [ ] **Bước 5:** Commit.

### Task 6: Chạy ma trận đầy đủ

- [ ] **Bước 1:** `uv run python -m begad_norm.normative.run_normative
  --config src/begad_norm/normative/config.yaml --workers 8`
- [ ] **Bước 2:** Kiểm chứng: đúng 300 dòng detail; 300 job completed trong `run_log.csv`
  (không có lỗi); provenance được ghi; kiểm tra ngẫu nhiên 2 run dir — best AUC trong
  `summary.json` bằng AUC tính lại từ `predictions.csv`.
- [ ] **Bước 3:** `uv run python -m begad_norm.baseline.aggregate_results --detail
  outputs/normative/results/results_detail.csv --summary
  outputs/normative/results/summary.csv --expected-rows 300 --per-group 100`
- [ ] **Bước 4:** Commit kết quả (file dữ liệu).

### Task 7: Báo cáo so sánh (EN + VI)

**Files:**
- Tạo: `docs/normative/normative_vs_baseline.md`, `docs/normative/normative_vs_baseline_vi.md`

- [ ] **Bước 1:** Viết báo cáo: mean ± std theo (bài toán, metric) cho normative_main so với
  baseline_balanced (từ hai summary CSV), cột Δ, ghi chú protocol, giới hạn (feature
  selection trong mẫu, AD n=16, một biến thể duy nhất, lưu ý determinism CPU, chỉ metric
  validation).
- [ ] **Bước 2:** Rà soát: mọi con số trong báo cáo truy được về dòng CSV; hai bản ngôn ngữ
  đồng bộ.
- [ ] **Bước 3:** Commit.

### Ngoài phạm vi (tường minh)

Các biến thể pooling/deviation khác, bộ 6 task kích thích, chỉnh ngưỡng, test set, hình so
sánh (chỉ khi được yêu cầu — sẽ đặt tại `Figure/code` + `Figure/outputs`).
