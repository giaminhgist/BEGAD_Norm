"""Write the 37-feature summary (definitions + task x contrast p-value table), EN and VI.

Input: posthoc_task_stratified/task_level_union_conditions_strict.csv and the cached report tables.
    uv run python -m begad_norm.feature_discovery.posthoc_feature_doc --run-dir outputs/feature_discovery/<run>
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ["KD1", "KD2", "KD3", "READING"]
CONTRASTS = [("CN_MCI", "CN vs MCI"), ("CN_AD", "CN vs AD"), ("MCI_AD", "MCI vs AD")]

# Plain-language meaning (EN, VI). Formulas come verbatim from feature_registry.csv.
MEANING = {
    "A002": ("Total time spent fixating during the task.", "Tổng thời gian nhìn cố định (fixation) trong task."),
    "A003": ("Number of fixations in the task.", "Số lần fixation trong task."),
    "A021": ("Number of saccade events in the task.", "Số saccade (cú chuyển mắt nhanh) trong task."),
    "A023": ("Typical saccade size.", "Độ lớn điển hình của saccade (trung vị biên độ)."),
    "A024": ("Spread of saccade sizes.", "Độ phân tán độ lớn saccade (IQR biên độ)."),
    "A026": ("Typical saccade duration.", "Thời lượng điển hình của một saccade."),
    "A027": ("Spread of saccade durations.", "Độ phân tán thời lượng saccade (IQR)."),
    "A028": ("Typical average saccade speed (not peak velocity).",
             "Tốc độ trung bình điển hình của saccade (không phải vận tốc đỉnh)."),
    "A029": ("Share of very small saccades (<1 degree).", "Tỉ lệ saccade rất nhỏ (<1 độ)."),
    "A030": ("Share of long saccades (>2 degrees).", "Tỉ lệ saccade dài (>2 độ)."),
    "A031": ("Total eye-movement path length in degrees.", "Tổng quãng đường mắt di chuyển (độ)."),
    "A033": ("How widely fixations spread over the screen.", "Mức độ phân tán của các fixation trên màn hình."),
    "A035": ("How evenly fixations cover an 8x6 screen grid.", "Mức độ phủ đều của fixation trên lưới 8x6 của màn hình."),
    "A046": ("Typical absolute pupil size (sensitive to lighting).",
             "Kích thước đồng tử tuyệt đối điển hình (nhạy với ánh sáng)."),
    "D007": ("Average number of fixations per target (digit/word).",
             "Số fixation trung bình trên mỗi target (chữ số/từ)."),
    "D010": ("Typical pupil change from first fixation on a target to the landing after leaving it.",
             "Thay đổi đồng tử điển hình từ fixation đầu trên target tới fixation hạ cánh sau khi rời target."),
    "D011": ("Typical within-target range of left pupil size.", "Biên độ dao động đồng tử trái điển hình trong một target."),
    "D015": ("Within-person link between how long a target is spoken and how many fixations it gets.",
             "Tương quan trong cùng người giữa thời gian phát âm một target và số fixation trên target đó."),
    "D017": ("Whether consecutive fixation durations resemble each other (serial dependence).",
             "Mức độ thời lượng của các fixation liên tiếp giống nhau (phụ thuộc chuỗi)."),
    "D018": ("Relative (scale-free) spread of fixation durations.", "Độ phân tán tương đối (không phụ thuộc thang đo) của thời lượng fixation."),
    "D021": ("Typical gap between one fixation ending and the next starting (movement time).",
             "Khoảng thời gian điển hình giữa hai fixation liên tiếp (thời gian chuyển mắt)."),
    "D022": ("Regular vs bursty timing of fixation onsets.", "Nhịp fixation đều hay dồn cụm (burstiness)."),
    "D023": ("Time from the first to the last fixation of the task (gaze-engaged duration).",
             "Thời gian từ fixation đầu tới fixation cuối của task (thời gian mắt tham gia làm task)."),
    "D024": ("Share of engaged time spent moving the eyes rather than fixating.",
             "Tỉ lệ thời gian dành cho chuyển mắt so với nhìn cố định."),
    "D025": ("Extra saccade duration per extra degree of amplitude (main-sequence slope).",
             "Thời lượng saccade tăng thêm cho mỗi độ biên độ (độ dốc main sequence)."),
    "D026": ("How tightly saccade duration follows amplitude.", "Mức độ chặt chẽ giữa thời lượng và biên độ saccade."),
    "D028": ("Share of eye movements going leftward (regressions and line returns together).",
             "Tỉ lệ chuyển mắt sang trái (gồm cả quay lại và xuống dòng)."),
    "D029": ("Typical size of forward (rightward) saccades.", "Độ lớn điển hình của saccade tiến (sang phải)."),
    "D030": ("Typical size of backward (leftward) saccades.", "Độ lớn điển hình của saccade lùi (sang trái)."),
    "D034": ("Whether consecutive saccade sizes resemble each other.", "Mức độ độ lớn các saccade liên tiếp giống nhau."),
    "D035": ("Link between fixation duration and the size of the next saccade (planning).",
             "Tương quan giữa thời lượng fixation và độ lớn saccade tiếp theo (lập kế hoạch)."),
    "D036": ("Link between the size of the incoming saccade and the following fixation duration.",
             "Tương quan giữa độ lớn saccade vừa tới và thời lượng fixation sau đó."),
    "D037": ("Fixation duration after leftward vs rightward movements (cost of going back).",
             "Thời lượng fixation sau chuyển mắt sang trái so với sang phải (chi phí quay lại)."),
    "D039": ("Horizontal vs vertical spread of fixations.", "Độ trải ngang so với độ trải dọc của fixation."),
    "D040": ("How consistently gaze moves from top to bottom over time.",
             "Mức độ mắt di chuyển đều từ trên xuống dưới theo thời gian."),
    "D041": ("Share of fixation time on the screen but outside the stimulus image.",
             "Tỉ lệ thời gian fixation trên màn hình nhưng nằm ngoài hình kích thích."),
    "D047": ("Variability of eye-to-screen distance (head/posture stability).",
             "Độ dao động khoảng cách mắt–màn hình (độ ổn định tư thế đầu)."),
}


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "NA"
    s = f"{p:.1e}" if p < 0.001 else f"{p:.3f}"
    if p < 0.05:
        return f"**{s}**"
    if p < 0.10:
        return f"*{s}*"
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    out = run / "posthoc_task_stratified"
    sel = pd.read_csv(out / "task_level_union_conditions_strict.csv")
    reg = pd.read_csv(run / "feature_registry.csv", keep_default_na=False).set_index("feature_id")
    mac = pickle.loads((run / "cache" / "report.pkl").read_bytes())["mac"]
    sel["_c"] = sel["cohort"].map({"RAW157": 0, "ROI64_T1235": 1})
    sel = sel.sort_values(["_c", "feature_id"]).reset_index(drop=True)
    missing = set(sel.feature_id) - set(MEANING)
    if missing:
        raise ValueError(f"meaning missing for {missing}")

    def cond_str(r, lang):
        c = [str(i) for i, k in ((1, "cond1"), (2, "cond2"), (3, "cond3")) if r[k]]
        return ", ".join(c)

    # p table rows
    prow, nrow = [], []
    for r in sel.itertuples():
        d = mac[(mac.cohort_id == r.cohort) & (mac.feature_id == r.feature_id)].set_index("scope_or_region_key")
        cells, ns = [], []
        for t in TASKS:
            for c, _ in CONTRASTS:
                if t in d.index and bool(d.loc[t, f"eligible_{c}"]):
                    cells.append(fmt_p(float(d.loc[t, f"p_{c}"])))
                else:
                    cells.append("NA")
            if t in d.index:
                x = d.loc[t]
                ns.append(f"{int(x['n1_CN_AD'])}/{int(x['n2_CN_MCI'])}/{int(x['n2_CN_AD'])}")
            else:
                ns.append("NA")
        prow.append(f"| {r.feature_id} | {r.name} | " + " | ".join(cells) + " |")
        nrow.append(f"| {r.feature_id} | " + " | ".join(ns) + " |")
    hdr1 = "| ID | Feature | " + " | ".join(f"{t} {lab}" for t in TASKS for _, lab in CONTRASTS) + " |"
    sep1 = "|---|---|" + "---|" * 12

    for lang in ("en", "vi"):
        L = []
        if lang == "en":
            L += ["# 37 task-level features: definitions and p-values", "",
                  f"Source run: `{run.name}` (freeze in `candidate_freeze.json`). Cohorts: **RAW157** = raw Tobii event "
                  "exports, 157 baseline participants (CN 100 / MCI 41 / AD 16); **ROI64_T1235** = original ROI workbook, "
                  "64 participants (CN 32 / MCI 17 / AD 15). The two cohorts are never pooled.", "",
                  "Selection rule (post hoc, requested after results were seen): a feature is listed if it meets at least "
                  "one of (1) at least 2 tasks with p < 0.10 for both CN vs MCI and CN vs AD; (2) at least 1 task with "
                  "MCI vs AD p < 0.10; (3) at least 1 task with any pairwise p < 0.05. 37 of 65 testable task-level "
                  "features qualify; all 37 meet condition 3, which by itself is expected for roughly half of features "
                  "under no true difference (12 tests per feature). The list is exploratory, not confirmed.", "",
                  "Tests: two-sided Mann-Whitney U (asymptotic, tie and continuity corrected), one value per participant "
                  "and task. p values are raw (uncorrected). Durations in seconds, amplitudes in degrees, pupil in mm.", "",
                  "## 1. Definitions", ""]
        else:
            L += ["# 37 feature mức task: định nghĩa và giá trị p", "",
                  f"Run nguồn: `{run.name}` (freeze trong `candidate_freeze.json`). Cohort: **RAW157** = export sự kiện "
                  "Tobii thô, 157 người baseline (CN 100 / MCI 41 / AD 16); **ROI64_T1235** = workbook ROI gốc, 64 người "
                  "(CN 32 / MCI 17 / AD 15). Hai cohort không bao giờ được gộp.", "",
                  "Luật chọn (post hoc, đặt ra sau khi đã xem kết quả): feature được liệt kê nếu thoả ít nhất một trong "
                  "(1) ít nhất 2 task có p < 0.10 ở cả CN vs MCI và CN vs AD; (2) ít nhất 1 task có p MCI vs AD < 0.10; "
                  "(3) ít nhất 1 task có p < 0.05 ở bất kỳ cặp nào. 37 trên 65 feature mức task kiểm định được đạt; cả 37 "
                  "đều thoả điều kiện 3, mà riêng điều kiện này đã có thể đúng với khoảng một nửa số feature khi không có "
                  "khác biệt thật (12 phép test mỗi feature). Danh sách mang tính thăm dò, chưa được xác nhận.", "",
                  "Kiểm định: Mann-Whitney U hai phía (xấp xỉ tiệm cận, hiệu chỉnh ties và continuity), mỗi người một giá "
                  "trị cho mỗi task. p là p thô (chưa hiệu chỉnh đa kiểm định). Thời lượng tính bằng giây, biên độ bằng độ, "
                  "đồng tử bằng mm.", "",
                  "## 1. Định nghĩa", ""]
        hdr = ("| ID | Feature | Cohort | Origin | Meaning | Formula | Unit | Conditions met |" if lang == "en" else
               "| ID | Feature | Cohort | Nguồn gốc | Ý nghĩa | Công thức | Đơn vị | Điều kiện thoả |")
        L += [hdr, "|---|---|---|---|---|---|---|---|"]
        for r in sel.itertuples():
            g = reg.loc[r.feature_id]
            mean = MEANING[r.feature_id][0 if lang == "en" else 1]
            formula = str(g["formula"]).replace("|", "/")
            L.append(f"| {r.feature_id} | `{r.name}` | {r.cohort} | {r.origin} | {mean} | {formula} | {g['unit']} | "
                     f"{cond_str(r._asdict(), lang)} |")
        notes_en = [
            "", "Notes on sources and definitions:",
            "- RAW157 fixation and saccade events are rebuilt from the onset and final export rows; offset = onset + "
            "duration. A 'gap-free transition' joins two consecutive fixations of the same task whose interval is covered "
            "by saccade events up to 10 ms; gaps are never bridged. A 'matched saccade' is a gap-free transition with "
            "exactly one saccade event, so amplitude and duration belong to the same movement.",
            "- Amplitude is the Tobii fixation-to-fixation angle (verified r = 0.99 against geometry). Direction "
            "(left/right) comes from fixation positions on screen; horizontal-dominant means |dx| > |dy|.",
            "- Pupil in RAW157 exists only at event boundary samples, not as a continuous trace; lighting was not "
            "controlled.",
            "- ROI64 features (D007, D010, D011, D015) summarise per-ROI workbook values over the targets of a task, "
            "requiring valid values for at least 60% of expected targets (couplings: at least 10 targets). D010 and D011 "
            "follow the export's Table II pupil convention (the 'exit' value is the landing fixation after leaving the "
            "target).",
            "- Full support rules, zero/missing rules and evidence: `feature_registry.csv`, "
            "`DATASET_PROPOSED_100_FEATURES.md`, `semantic_audit.md`."]
        notes_vi = [
            "", "Ghi chú về nguồn và định nghĩa:",
            "- Sự kiện fixation và saccade của RAW157 được dựng lại từ dòng onset và dòng final của export; offset = "
            "onset + thời lượng. 'Chuyển tiếp không gián đoạn' nối hai fixation liên tiếp trong cùng task mà khoảng giữa "
            "được các saccade phủ kín (sai lệch tối đa 10 ms); không bao giờ nối qua chỗ mất tín hiệu. 'Saccade khớp' là "
            "chuyển tiếp không gián đoạn có đúng một saccade, nên biên độ và thời lượng thuộc cùng một cú chuyển mắt.",
            "- Biên độ là góc fixation-tới-fixation do Tobii tính (đã kiểm tra r = 0.99 với hình học). Hướng trái/phải "
            "lấy từ vị trí fixation trên màn hình; 'chủ yếu theo chiều ngang' nghĩa là |dx| > |dy|.",
            "- Đồng tử trong RAW157 chỉ có tại các mẫu ở biên sự kiện, không phải chuỗi liên tục; ánh sáng không được "
            "kiểm soát.",
            "- Các feature ROI64 (D007, D010, D011, D015) tóm tắt giá trị theo ROI của workbook trên các target của một "
            "task, yêu cầu có giá trị hợp lệ ở ít nhất 60% target dự kiến (với tương quan: ít nhất 10 target). D010 và "
            "D011 theo quy ước đồng tử của Bảng II (giá trị 'ra' là fixation hạ cánh sau khi rời target).",
            "- Luật hỗ trợ, luật zero/missing và bằng chứng đầy đủ: `feature_registry.csv`, "
            "`DATASET_PROPOSED_100_FEATURES.md`, `semantic_audit.md`."]
        L += notes_en if lang == "en" else notes_vi
        if lang == "en":
            L += ["", "## 2. p-values: 4 tasks x 3 comparisons", "",
                  "**Bold** = p < 0.05; *italic* = 0.05 <= p < 0.10; NA = not testable (fewer than 8 valid participants "
                  "or below 60% coverage in a group). Raw p values; see section 3 for group sizes.", ""]
        else:
            L += ["", "## 2. Giá trị p: 4 task x 3 phép so sánh", "",
                  "**In đậm** = p < 0.05; *in nghiêng* = 0.05 <= p < 0.10; NA = không kiểm định được (ít hơn 8 người hợp "
                  "lệ hoặc dưới 60% coverage trong một nhóm). p thô; cỡ mẫu xem mục 3.", ""]
        L += [hdr1, sep1] + prow
        if lang == "en":
            L += ["", "## 3. Valid participants per task (CN/MCI/AD)", ""]
        else:
            L += ["", "## 3. Số người hợp lệ theo task (CN/MCI/AD)", ""]
        L += ["| ID | " + " | ".join(TASKS) + " |", "|---|" + "---|" * 4] + nrow
        if lang == "en":
            L += ["", "## 4. Caveats", "",
                  "- p values are uncorrected. Within the pre-registered families, only D023@KD3, A002@KD3, A003@KD3 and "
                  "D023@KD2 reach joint BH q <= 0.10 (none survive BY).",
                  "- D023, A002, A003 and A021 are strongly correlated (longer reading of the card = more fixations and "
                  "saccades); they are not independent confirmations.",
                  "- The selection thresholds were chosen after inspecting results; any predictive use must repeat the "
                  "selection inside training data or use an independent cohort.",
                  "- AD groups are small (15-16), and no demographic adjustment was possible (no age/sex/education data)."]
        else:
            L += ["", "## 4. Lưu ý", "",
                  "- p chưa hiệu chỉnh. Trong các họ đa kiểm định đã đăng ký trước, chỉ D023@KD3, A002@KD3, A003@KD3 và "
                  "D023@KD2 đạt joint BH q <= 0.10 (không cái nào còn lại với BY).",
                  "- D023, A002, A003 và A021 tương quan mạnh (đọc thẻ lâu hơn = nhiều fixation và saccade hơn); chúng "
                  "không phải các bằng chứng độc lập.",
                  "- Ngưỡng chọn được đặt sau khi đã xem kết quả; mọi ứng dụng dự đoán phải lặp lại việc chọn feature bên "
                  "trong dữ liệu train hoặc dùng cohort độc lập.",
                  "- Nhóm AD nhỏ (15-16 người) và không thể hiệu chỉnh nhân khẩu học (không có dữ liệu tuổi/giới/học vấn)."]
        name = "FEATURES_37_SUMMARY" + ("" if lang == "en" else ".vi") + ".md"
        (out / name).write_text("\n".join(L) + "\n", encoding="utf-8")
        print("wrote", out / name)


if __name__ == "__main__":
    main()
