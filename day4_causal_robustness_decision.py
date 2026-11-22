"""
اليوم الرابع: الاستدلال السببي والمتانة والقرار النهائي
=============================================================
المشروع : تقييم أثر المساعد الذكي (AI Bot) على زمن حل التذاكر في الدعم الفني
الدورة  : التجارب واختبارات A/B والاستدلال السببي (أكاديمية سدايا)

محتوى الملف:
  1) رسم DAG يوضح لماذا تكفي المقارنة المباشرة في التجربة العشوائية، ومكان المتغير الوسيط (حل المساعد للتذكرة).
  2) انحدار معدّل بأخطاء معيارية عنقودية حسب الموظف.
  3) اختلاف الأثر حسب نوع التذكرة: اختبار تفاعل (Wald) + أثر كل نوع مع Holm.
  4) تحليلات الحساسية: حذف القيم المتطرفة، حذف الأسبوع الأول، التحليل على مستوى الموظف،
     مقياس اللوغاريتم، واختبار التبديل العشوائي (Permutation) داخل كل موظف.
  5) فحص تأثير الجِدّة أسبوعياً.
  6) القرار النهائي بقاعدة صريحة + أثر تجاري توضيحي.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as tdist

from utils import CHARTS, DATA, RESULTS, coef_ci, fmt_p, holm, ols_cluster, pct, wald_test, welch

CATS = ["Account", "Software", "Network", "Hardware"]


def design(df: pd.DataFrame, interact: bool = False) -> pd.DataFrame:
    X = pd.DataFrame({"const": 1.0, "treat": df.treatment.astype(float)}, index=df.index)
    for k in CATS[1:]:
        X[f"cat_{k}"] = (df.category == k).astype(float)
    X["cat_Unknown"] = (df.category == "Unknown").astype(float)
    for k in ("Low", "High"):
        X[f"prio_{k}"] = (df.priority == k).astype(float)
    X["complexity"] = df.complexity
    X["agent_pre"] = df.agent_pre_avg_hours
    if interact:
        for k in CATS[1:]:
            X[f"treat_x_{k}"] = X["treat"] * X[f"cat_{k}"]
    return X


def adjusted_effect(df: pd.DataFrame) -> dict:
    X = design(df)
    beta, V, dof = ols_cluster(X.values, df.resolution_hours.values, df.agent_id.values)
    out = coef_ci(beta, V, dof, list(X.columns).index("treat"))
    base = df[df.treatment == 0].resolution_hours.mean()
    out.update({"pct": out["coef"] / base * 100, "pct_low": out["ci_low"] / base * 100, "pct_high": out["ci_high"] / base * 100})
    return out


def draw_dag(ax):
    nodes = {"Randomization\n(random 50/50)": (1.2, 5.3), "AI bot first\n(T)": (1.2, 3.0),
             "Bot resolves\nthe ticket (mediator)": (4.7, 4.6), "Resolution time\n(Y)": (8.2, 3.0),
             "Category, priority,\ncomplexity": (7.2, 5.9), "Agent skill": (4.7, 0.9),
             "Reopen / CSAT\n(guardrails)": (8.4, 1.0)}
    edges = [("Randomization\n(random 50/50)", "AI bot first\n(T)"), ("AI bot first\n(T)", "Bot resolves\nthe ticket (mediator)"),
             ("Bot resolves\nthe ticket (mediator)", "Resolution time\n(Y)"), ("AI bot first\n(T)", "Resolution time\n(Y)"),
             ("Category, priority,\ncomplexity", "Resolution time\n(Y)"), ("Category, priority,\ncomplexity", "Bot resolves\nthe ticket (mediator)"),
             ("Agent skill", "Resolution time\n(Y)"), ("AI bot first\n(T)", "Reopen / CSAT\n(guardrails)")]
    for name, (x, y) in nodes.items():
        ax.text(x, y, name, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="#eef6fa", ec="#1f6f8b"))
    for a, b in edges:
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.4, shrinkA=32, shrinkB=38, color="#333"))
    ax.text(4.7, -0.5, "Randomization is the only arrow into T, so ticket category / priority / agent skill cannot confound T -> Y.\n"
            "Adjusting for them only improves precision. The bot-resolution path is part of the effect (do not condition on it).",
            ha="center", fontsize=9)
    ax.set(xlim=(0, 10), ylim=(-1.2, 6.6), title="Day 4 - Causal diagram (DAG) of the randomized experiment"); ax.axis("off")


def main() -> None:
    df = pd.read_csv(DATA / "dataset_clean.csv")
    df["treatment"] = (df.variant == "B").astype(int)
    df["log_h"] = np.log(df.resolution_hours)
    ab = json.loads((RESULTS / "ab_results.json").read_text(encoding="utf-8"))
    base = df[df.treatment == 0].resolution_hours.mean()
    out: dict = {}

    adj = adjusted_effect(df)
    out["adjusted"] = adj
    print(f"[انحدار معدّل + عنقودي] {adj['coef']:+.2f} ساعة = {adj['pct']:+.1f}% (CI95% [{adj['pct_low']:.1f}%, {adj['pct_high']:.1f}%]) p = {adj['p_value']:.2g}")

    sub = df[df.category != "Unknown"]
    Xi = design(sub, interact=True)
    beta, V, _ = ols_cluster(Xi.values, sub.resolution_hours.values, sub.agent_id.values)
    idx = [list(Xi.columns).index(f"treat_x_{k}") for k in CATS[1:]]
    chi, p_int = wald_test(beta, V, idx)
    out["interaction_test"] = {"wald_chi2": chi, "df": len(idx), "p_value": p_int}
    print(f"[تفاعل المعالجة × النوع] Wald χ²({len(idx)}) = {chi:.2f}, p = {p_int:.4g}")
    cat_rows = []
    for k in CATS:
        s = sub[sub.category == k]
        X = pd.DataFrame({"const": 1.0, "treat": s.treatment.astype(float), "complexity": s.complexity, "agent_pre": s.agent_pre_avg_hours,
                          "prio_Low": (s.priority == "Low").astype(float), "prio_High": (s.priority == "High").astype(float)})
        bb, vv, dof = ols_cluster(X.values, s.resolution_hours.values, s.agent_id.values)
        rr = coef_ci(bb, vv, dof, 1)
        b0 = s[s.treatment == 0].resolution_hours.mean()
        rr.update({"category": k, "hours": rr["coef"], "pct": rr["coef"] / b0 * 100, "pct_low": rr["ci_low"] / b0 * 100,
                   "pct_high": rr["ci_high"] / b0 * 100, "n": int(len(s)),
                   "bot_resolved_share_B": float(s[s.treatment == 1].bot_resolved.mean())})
        cat_rows.append(rr)
    for rr, p_ in zip(cat_rows, holm([x["p_value"] for x in cat_rows])):
        rr["p_holm"] = p_
    out["by_category"] = cat_rows
    pd.DataFrame(cat_rows).to_csv(RESULTS / "effect_by_category.csv", index=False)
    for rr in cat_rows:
        print(f"  {rr['category']:<9} {rr['pct']:+.1f}% [{rr['pct_low']:.1f}, {rr['pct_high']:.1f}] | حل المساعد {rr['bot_resolved_share_B']:.0%} | p Holm = {fmt_p(rr['p_holm'])}")

    sens = [("Adjusted regression (main)", adj["pct"], adj["pct_low"], adj["pct_high"])]
    cut = df[df.resolution_hours <= df.resolution_hours.quantile(0.99)]
    a2 = adjusted_effect(cut); sens.append(("Drop top 1% longest tickets", a2["pct"], a2["pct_low"], a2["pct_high"]))
    a3 = adjusted_effect(df[df.day > 7]); sens.append(("Drop first week (learning)", a3["pct"], a3["pct_low"], a3["pct_high"]))
    ag = df.groupby(["agent_id", "treatment"]).resolution_hours.mean().unstack().dropna()
    dif = (ag[1] - ag[0]).values
    se = dif.std(ddof=1) / np.sqrt(len(dif)); tc = tdist.ppf(0.975, len(dif) - 1)
    sens.append(("Agent-level paired analysis", dif.mean() / base * 100, (dif.mean() - tc * se) / base * 100, (dif.mean() + tc * se) / base * 100))
    wl = welch(df[df.treatment == 0].log_h, df[df.treatment == 1].log_h)
    sens.append(("Log scale (geometric ratio)", pct(wl["diff"]), pct(wl["ci_low"]), pct(wl["ci_high"])))

    rng = np.random.default_rng(3)
    groups = [g.index.values for _, g in df.groupby("agent_id")]
    y, tr = df.resolution_hours.values, df.treatment.values
    obs = y[tr == 1].mean() - y[tr == 0].mean()
    perm = np.empty(2000)
    for i in range(2000):
        tp = tr.copy()
        for gi in groups:
            tp[gi] = rng.permutation(tr[gi])
        perm[i] = y[tp == 1].mean() - y[tp == 0].mean()
    p_perm = float((np.sum(np.abs(perm) >= abs(obs)) + 1) / (len(perm) + 1))
    out["permutation"] = {"observed_diff_hours": float(obs), "p_value": p_perm}
    out["sensitivity"] = [{"name": n, "pct": p, "pct_low": lo, "pct_high": hi} for n, p, lo, hi in sens]
    print("--- الحساسية ---")
    for n, p, lo, hi in sens:
        print(f"  {n:<32} {p:+.1f}% [{lo:.1f}, {hi:.1f}]")
    print(f"  اختبار التبديل داخل الموظف: p = {p_perm:.4f}")
    print("  (مقياس اللوغاريتم أكبر لأن التذاكر التي يحلها المساعد سريعة جداً؛ لذلك المقياس الرئيسي هو المتوسط الخام)")

    wk = []
    for w in range(4):
        s = df[(df.day > 7 * w) & (df.day <= 7 * (w + 1))]
        r_ = welch(s[s.treatment == 0].resolution_hours, s[s.treatment == 1].resolution_hours)
        b0 = s[s.treatment == 0].resolution_hours.mean()
        wk.append((w + 1, r_["diff"] / b0 * 100, r_["ci_low"] / b0 * 100, r_["ci_high"] / b0 * 100))
    wk = pd.DataFrame(wk, columns=["week", "pct", "lo", "hi"])
    slope = float(np.polyfit(wk.week, wk.pct, 1)[0])
    out["novelty"] = {"weekly_pct": wk.pct.round(2).tolist(), "slope_pct_per_week": slope}
    print(f"[الجِدّة] الأثر الأسبوعي {wk.pct.round(1).tolist()} (ميل {slope:+.2f} نقطة/أسبوع)")

    # القرار (قاعدة معرّفة صراحةً)
    guards = ab["guardrails"]
    sig_neg = (ab["cuped"]["pct_high"] < 0 and adj["pct_high"] < 0 and sens[3][3] < 0 and p_perm < 0.05)
    harmed = [g for g in guards if (g["diff"] >= g["margin"] if g["name"].startswith("Reopen") else g["diff"] <= g["margin"])]
    unproven = [g for g in guards if g["status"] == "FAIL" and g not in harmed]
    if not sig_neg:
        decision = "النتيجة غير حاسمة"
    elif harmed:
        decision = "عدم الإطلاق"
    elif unproven:
        decision = "إطلاق تدريجي مع مراقبة مقاييس الحماية"
    else:
        decision = "إطلاق"
    strong = [x["category"] for x in cat_rows if x["pct_high"] < -3]           # فائدة مثبتة
    harm = [x["category"] for x in cat_rows if x["pct_low"] > 0]               # ارتفاع الزمن بدلالة (احتمال ضرر)
    weak = [x["category"] for x in cat_rows if x["category"] not in strong + harm]   # غير مؤكد
    if decision == "إطلاق" and (weak or harm):
        parts = [f"إطلاق للأنواع ذات الفائدة المثبتة ({'، '.join(strong)})"]
        if weak:
            parts.append(f"وإعادة تقييم ({'، '.join(weak)})")
        if harm:
            parts.append(f"واستثناء ({'، '.join(harm)}) لظهور ارتفاع في زمن الحل")
        decision = " ".join(parts)
    out["decision"] = decision
    saved = ab["primary"]["mean_control"] - ab["primary"]["mean_treatment"]
    out["illustrative_impact"] = {"hours_saved_per_ticket": saved, "tickets_per_year": 250 * 365, "hours_saved_per_year": saved * 250 * 365}
    (RESULTS / "robustness_and_decision.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n>>> القرار: {decision}")

    pr = ab["primary"]
    g_lines = "\n".join(f"- {g['name']}: الفرق {g['diff']:+.4f} | الحد {'الأعلى' if g['name'].startswith('Reopen') else 'الأدنى'} "
                        f"{g['ci_high'] if g['name'].startswith('Reopen') else g['ci_low']:.4f} | الهامش {g['margin']} => {g['status']}" for g in guards)
    md = f"""# التوصية النهائية

**القرار: {decision}**

## الأدلة على الأثر (متوسط زمن الحل)
- A/B مباشر (Welch): {pr['mean_control']:.2f} -> {pr['mean_treatment']:.2f} ساعة، أي {pr['pct_change']:+.1f}% (CI95% [{pr['pct_low']:.1f}%, {pr['pct_high']:.1f}%])
- CUPED: {ab['cuped']['pct_change']:+.1f}% | انحدار معدّل عنقودي: {adj['pct']:+.1f}% | على مستوى الموظف: {sens[3][1]:+.1f}%
- اختبار التبديل داخل الموظف: p = {p_perm:.4f}
- الأثر يتفاوت حسب النوع (p التفاعل = {p_int:.2g}): فائدة واضحة في {', '.join(strong) or '-'}؛ غير مؤكدة في {', '.join(weak) or '-'}؛ ارتفاع في الزمن في {', '.join(harm) or '-'}.
- ملاحظة: حدّ فترة الثقة الأضعف ({pr['pct_high']:.1f}%) أقل من أصغر أثر مخطط (10%)، فالأثر مؤكد لكن حجمه الدقيق غير محسوم.

## مقاييس الحماية
{g_lines}

## قاعدة القرار (معرّفة مسبقاً)
إطلاق إذا ثبت انخفاض الزمن بكل الطرق ونجحت مقاييس الحماية؛ تدريجي إذا لم يثبت عدم التدهور في مقياس حماية؛
عدم إطلاق إذا ظهر تدهور فعلي؛ غير حاسم إذا لم يثبت الأثر.

## أثر تجاري توضيحي (افتراضات محاكاة)
توفير ≈ {saved:.2f} ساعة لكل تذكرة؛ ومع 250 تذكرة يومياً ≈ {saved*250*365:,.0f} ساعة انتظار سنوياً.
"""
    (RESULTS / "final_recommendation.md").write_text(md, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11, 7)); draw_dag(ax); fig.tight_layout()
    fig.savefig(CHARTS / "day4_dag.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    for i, x in enumerate(cat_rows):
        ax[0].errorbar(x["pct"], i, xerr=[[x["pct"] - x["pct_low"]], [x["pct_high"] - x["pct"]]], fmt="s", capsize=5, color="#c60")
        ax[0].text(x["pct_high"] + 1.5, i, f"p(Holm)={fmt_p(x['p_holm'])}", va="center", fontsize=8)
    ax[0].set_yticks(range(4)); ax[0].set_yticklabels(CATS); ax[0].invert_yaxis(); ax[0].axvline(0, color="k", ls="--")
    ax[0].set(title=f"Effect by ticket category (interaction p = {fmt_p(p_int)})", xlabel="% change in mean resolution time", xlim=(-60, 30))
    for i, (n_, p_, lo, hi) in enumerate(sens):
        ax[1].errorbar(p_, i, xerr=[[p_ - lo], [hi - p_]], fmt="o", capsize=5, color="#2a6")
    ax[1].set_yticks(range(len(sens))); ax[1].set_yticklabels([s[0] for s in sens], fontsize=8); ax[1].invert_yaxis()
    ax[1].axvline(0, color="k", ls="--"); ax[1].set(title=f"Sensitivity analyses (permutation p = {fmt_p(p_perm)})", xlabel="% change")
    ax[2].errorbar(wk.week, wk.pct, yerr=[wk.pct - wk.lo, wk.hi - wk.pct], fmt="o-", capsize=5, color="#d98324")
    ax[2].axhline(0, color="k", ls="--"); ax[2].set(title="Effect by week (novelty check)", xlabel="Week", ylabel="% change", xticks=[1, 2, 3, 4])
    fig.suptitle("Day 4 - Heterogeneity, robustness and novelty", fontsize=14)
    fig.tight_layout(); fig.savefig(CHARTS / "day4_robustness.png", dpi=150); plt.close(fig)
    print("تم حفظ الرسوم وتوصية outputs/results/final_recommendation.md")


if __name__ == "__main__":
    main()
