"""
اليوم الثاني: البيانات وفحوص الجودة
=============================================================
المشروع : تقييم أثر المساعد الذكي (AI Bot) على زمن حل التذاكر في الدعم الفني
الدورة  : التجارب واختبارات A/B والاستدلال السببي (أكاديمية سدايا)

البيانات محاكاة (Simulated) عبر simulation.py. ندخل عيوباً مقصودة (سجلات مكررة وفئات مفقودة)
لإظهار فحوص الجودة، ونُجري فحص SRM على البيانات السليمة وعلى نسخة بها خلل تسجيل مقصود.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

from simulation import CONFIG, simulate_experiment, srm_check
from utils import CHARTS, DATA, RESULTS

DURATION_DAYS = 28


def smd(a, b) -> float:
    return float((b.mean() - a.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))


def main() -> None:
    print("--- بدء محاكاة وتجهيز بيانات مشروع A/B Testing ---")
    data = simulate_experiment(CONFIG, duration_days=DURATION_DAYS)
    rng = np.random.default_rng(CONFIG["seed"] + 1)
    raw = pd.concat([data, data.sample(15, random_state=CONFIG["seed"])], ignore_index=True)   # تكرار مقصود
    raw.loc[rng.choice(len(raw), 12, replace=False), "category"] = np.nan                     # مفقود مقصود
    raw = raw.sample(frac=1, random_state=CONFIG["seed"]).reset_index(drop=True)
    raw.to_csv(DATA / "dataset.csv", index=False)                       # تلقائياً توليد البيانات وحفظها في مجلد data
    print(f"تم توليد البيانات بنجاح وحفظها في: data/dataset.csv\nعدد السجلات الإجمالي: {len(raw)}")

    rep = {"raw_rows": int(len(raw)), "duplicates": int(raw.duplicated(subset="ticket_id").sum())}
    df = raw.drop_duplicates(subset="ticket_id").copy()
    miss = raw.isna().sum()
    rep["missing"] = {k: int(v) for k, v in miss[miss > 0].items()}
    df["category"] = df["category"].fillna("Unknown")
    print(f"[تكرار] {rep['duplicates']} سجلاً حُذف | [مفقود] {rep['missing']} (csat مفقود بطبيعته: لم يُقيّم العميل)")
    rep["invalid_values"] = int((df.resolution_hours <= 0).sum() + (df.first_response_min <= 0).sum())

    counts = df["variant"].value_counts()
    rep["srm"] = {"A": int(counts.get("A", 0)), "B": int(counts.get("B", 0)),
                  **srm_check(counts.get("A", 0), counts.get("B", 0), CONFIG["treatment_share"])}

    # عرض توضيحي: ماذا لو حدث خلل في تسجيل 5% من تذاكر B؟
    bug = simulate_experiment({**CONFIG, "srm_bug_share": 0.05}, DURATION_DAYS)
    bc = bug["variant"].value_counts()
    print("[عرض توضيحي لخلل التسجيل]", end=" ")
    rep["srm_bug_demo"] = {"A": int(bc["A"]), "B": int(bc["B"]),
                           **srm_check(bc["A"], bc["B"], CONFIG["treatment_share"])}

    a, b = df[df.variant == "A"], df[df.variant == "B"]
    bal = {"complexity_SMD": smd(a.complexity, b.complexity),
           "agent_pre_avg_hours_SMD": smd(a.agent_pre_avg_hours, b.agent_pre_avg_hours)}
    for col in ("category", "priority"):
        bal[f"{col}_chi2_p"] = float(chi2_contingency(pd.crosstab(df[col], df.variant))[1])
    rep["balance"] = {k: round(v, 4) for k, v in bal.items()}
    print(f"[توازن] {rep['balance']}")

    df.to_csv(DATA / "dataset_clean.csv", index=False)
    rep["clean_rows"] = int(len(df))
    (RESULTS / "data_quality.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax[0, 0].bar(["Control (A)", "Treatment (B)"], [counts["A"], counts["B"]], color=["#1f6f8b", "#d98324"])
    ax[0, 0].axhline(counts.sum() / 2, color="k", ls="--")
    ax[0, 0].set(title=f"SRM check (p = {rep['srm']['p_value']:.3f})", ylabel="Tickets")
    ax[0, 1].bar(["A", "B"], [bc["A"], bc["B"]], color=["#1f6f8b", "#a33"])
    ax[0, 1].axhline(bc.sum() / 2, color="k", ls="--")
    ax[0, 1].set(title=f"Demo: 5% of B lost -> SRM detected (p = {rep['srm_bug_demo']['p_value']:.1e})", ylabel="Tickets")
    ax[0, 2].bar(["Duplicates", "Missing category"], [rep["duplicates"], rep["missing"].get("category", 0)], color="#a33")
    ax[0, 2].set(title="Defects found in raw data", ylabel="Rows")
    smds = {"complexity": bal["complexity_SMD"], "agent pre-avg hours": bal["agent_pre_avg_hours_SMD"]}
    for c in CONFIG["categories"]:
        smds[c] = smd((a.category == c).astype(float), (b.category == c).astype(float))
    ax[1, 0].barh(list(smds), list(smds.values()), color="#4c9a2a")
    for x in (-0.1, 0.1):
        ax[1, 0].axvline(x, color="r", ls="--")
    ax[1, 0].set(title="Covariate balance (SMD, |0.1| threshold)", xlim=(-0.2, 0.2))
    bins = np.linspace(0, 60, 61)
    for g, name, col in ((a, "Control (A)", "#1f9a78"), (b, "Treatment (B)", "#e8744f")):
        ax[1, 1].hist(g.resolution_hours.clip(upper=60), bins=bins, density=True, alpha=0.55, label=name, color=col)
    ax[1, 1].set(title="Resolution time distribution (right-skewed, clipped at 60h)", xlabel="Hours", ylabel="Density")
    ax[1, 1].legend()
    ax[1, 2].bar(["Control (A)", "Treatment (B)"], [a.bot_resolved.mean() * 100, b.bot_resolved.mean() * 100], color=["#1f6f8b", "#d98324"])
    ax[1, 2].set(title="Tickets resolved by the AI bot (%)", ylabel="%")
    fig.suptitle("Day 2 - Data quality checks", fontsize=14)
    fig.tight_layout(); fig.savefig(CHARTS / "day2_data_quality.png", dpi=150); plt.close(fig)

    # الرسم الذي كان في مستودعك: توزيع زمن الحل (A مقابل B)
    fig, ax = plt.subplots(figsize=(10, 6))
    for g, name, col in ((a, "Control (A)", "#1f9a78"), (b, "Treatment (B)", "#e8744f")):
        ax.hist(g.resolution_hours.clip(upper=60), bins=bins, density=True, alpha=0.55, label=name, color=col)
    ax.set(title="Comparison of Resolution Time: Control vs AI Support (Treatment)",
           xlabel="Resolution time (hours, clipped at 60)", ylabel="Density")
    ax.legend(title="group")
    fig.tight_layout(); fig.savefig(CHARTS / "ab_test_resolution_time_distribution.png", dpi=150); plt.close(fig)
    print("--- انتهى تشخيص وتجهيز البيانات بنجاح ---")


if __name__ == "__main__":
    main()
