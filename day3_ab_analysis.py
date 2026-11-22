"""
اليوم الثالث: تحليل اختبار A/B
=============================================================
المشروع : تقييم أثر المساعد الذكي (AI Bot) على زمن حل التذاكر في الدعم الفني
الدورة  : التجارب واختبارات A/B والاستدلال السببي (أكاديمية سدايا)

الأساليب:
  - المقياس الرئيسي: متوسط زمن الحل (Welch + فترة ثقة 95%) مع Bootstrap كتحقق مستقل.
  - Mann-Whitney (اختبار لا معلمي لأن التوزيع منحرف) ومقارنة الوسيط.
  - CUPED باستخدام agent_pre_avg_hours (متوسط أداء الموظف قبل التجربة).
  - مقياس ثانوي: زمن أول رد. تصحيح Holm للمقياس الثانوي وشرائح الأولوية.
  - مقاييس الحماية بعدم التدهور: إعادة الفتح خلال 7 أيام، ورضا العملاء CSAT.
  - الأثر التراكمي يومياً والتحقق من تأثير الجِدّة.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from utils import CHARTS, DATA, RESULTS, bootstrap_pct_change, cuped, fmt_p, holm, prop_diff, welch

REOPEN_MARGIN, CSAT_MARGIN = 0.02, -0.20


def add_pct(r: dict, base: float) -> dict:
    """نسبة التغير مقدّرة بقسمة الفرق وحدود فترة الثقة على متوسط المقارنة."""
    r["pct_change"], r["pct_low"], r["pct_high"] = (r[k] / base * 100 for k in ("diff", "ci_low", "ci_high"))
    return r


def main() -> None:
    path = DATA / "dataset_clean.csv"
    if not path.exists():
        raise SystemExit("شغّل day2_data_and_quality_checks.py أولاً")
    df = pd.read_csv(path)
    df["treatment"] = (df.variant == "B").astype(int)
    a, b = df[df.treatment == 0], df[df.treatment == 1]
    base = a.resolution_hours.mean()
    res: dict = {}

    # 1) المقياس الرئيسي
    r = add_pct(welch(a.resolution_hours, b.resolution_hours), base)
    r["median_control"], r["median_treatment"] = float(a.resolution_hours.median()), float(b.resolution_hours.median())
    r["mannwhitney_p"] = float(mannwhitneyu(a.resolution_hours, b.resolution_hours).pvalue)
    boot = bootstrap_pct_change(a.resolution_hours, b.resolution_hours)
    r["bootstrap"] = boot
    res["primary"] = r
    print("=== المقياس الرئيسي: متوسط زمن حل التذكرة ===")
    print(f"A: {r['mean_control']:.2f} ساعة | B: {r['mean_treatment']:.2f} ساعة | الفرق {r['diff']:+.2f} ساعة "
          f"({r['pct_change']:+.1f}%، CI95% [{r['pct_low']:.1f}%, {r['pct_high']:.1f}%]) | p = {r['p_value']:.2g}")
    print(f"Bootstrap: {boot['point']:+.1f}% [{boot['ci_low']:.1f}%, {boot['ci_high']:.1f}%] | "
          f"الوسيط {r['median_control']:.2f} -> {r['median_treatment']:.2f} | Mann-Whitney p = {r['mannwhitney_p']:.2g}")

    # 2) CUPED
    y_adj, theta, vr = cuped(df.resolution_hours, df.agent_pre_avg_hours)
    df["y_cuped"] = y_adj
    cu = add_pct(welch(df[df.treatment == 0].y_cuped, df[df.treatment == 1].y_cuped), base)
    cu.update({"theta": theta, "variance_reduction_pct": vr * 100})
    res["cuped"] = cu
    print(f"CUPED: {cu['pct_change']:+.1f}% (CI95% [{cu['pct_low']:.1f}%, {cu['pct_high']:.1f}%]) | تقليل التباين {vr*100:.1f}%")

    # 3) وصف آلية الأثر (وصفي فقط) + المقياس الثانوي والشرائح
    res["mechanism"] = {"bot_resolved_share_B": float(b.bot_resolved.mean()),
                        "escalation_rate_B": float(b.escalated_to_human.mean()),
                        "mean_hours_bot_resolved": float(b[b.bot_resolved == 1].resolution_hours.mean()),
                        "mean_hours_escalated_B": float(b[b.bot_resolved == 0].resolution_hours.mean())}
    m = res["mechanism"]
    print(f"[الآلية - وصفي] حل المساعد {m['bot_resolved_share_B']:.1%} من تذاكر B وصُعِّد {m['escalation_rate_B']:.1%}")

    fr = welch(np.log(a.first_response_min), np.log(b.first_response_min))
    fr["pct_change"] = float((np.exp(fr["diff"]) - 1) * 100)
    segs = []
    for pr in ("Low", "Medium", "High"):
        s = df[df.priority == pr]
        w = add_pct(welch(s[s.treatment == 0].resolution_hours, s[s.treatment == 1].resolution_hours),
                    s[s.treatment == 0].resolution_hours.mean())
        segs.append({"segment": f"priority={pr}", **w})
    adj = holm([s["p_value"] for s in segs] + [fr["p_value"]])
    for s, p_ in zip(segs, adj):
        s["p_holm"] = p_
    fr["p_holm"] = adj[-1]
    res["first_response"], res["segments_by_priority"] = fr, segs
    print(f"زمن أول رد: {fr['pct_change']:+.1f}% (p Holm = {fmt_p(fr['p_holm'])})")
    for s in segs:
        print(f"  {s['segment']:<18} {s['pct_change']:+.1f}% | p Holm = {fmt_p(s['p_holm'])}")

    # 4) مقاييس الحماية
    g_re = prop_diff(a.reopened_7d, b.reopened_7d)
    g_re.update({"name": "Reopen rate (7d)", "margin": REOPEN_MARGIN,
                 "status": "PASS" if g_re["ci_high"] < REOPEN_MARGIN else "FAIL"})
    g_cs = welch(a.csat.dropna(), b.csat.dropna())
    g_cs.update({"name": "CSAT (1-5)", "margin": CSAT_MARGIN,
                 "status": "PASS" if g_cs["ci_low"] > CSAT_MARGIN else "FAIL"})
    res["guardrails"] = [g_re, g_cs]
    print("--- مقاييس الحماية ---")
    print(f"إعادة الفتح: فرق {g_re['diff']*100:+.2f} نقطة | الحد الأعلى {g_re['ci_high']*100:.2f} مقابل الهامش {REOPEN_MARGIN*100:.0f} -> {g_re['status']}")
    print(f"CSAT: فرق {g_cs['diff']:+.3f} | الحد الأدنى {g_cs['ci_low']:.3f} مقابل الهامش {CSAT_MARGIN} -> {g_cs['status']}")

    # 5) الأثر التراكمي + الجِدّة
    cum = []
    for d in range(3, int(df.day.max()) + 1):
        s = df[df.day <= d]
        w = welch(s[s.treatment == 0].resolution_hours, s[s.treatment == 1].resolution_hours)
        cum.append((d, *(w[k] / s[s.treatment == 0].resolution_hours.mean() * 100 for k in ("diff", "ci_low", "ci_high"))))
    cum = pd.DataFrame(cum, columns=["day", "pct", "lo", "hi"])
    early, late = df[df.day <= 7], df[df.day > 21]
    nov = {k: float(welch(s[s.treatment == 0].resolution_hours, s[s.treatment == 1].resolution_hours)["diff"]
                    / s[s.treatment == 0].resolution_hours.mean() * 100) for k, s in (("first_week_pct", early), ("last_week_pct", late))}
    res["novelty_check"] = nov
    print(f"[الجِدّة] الأسبوع الأول {nov['first_week_pct']:+.1f}% مقابل الأسبوع الأخير {nov['last_week_pct']:+.1f}%")

    (RESULTS / "ab_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(segs).to_csv(RESULTS / "effect_by_priority.csv", index=False)

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    for g, name, col in ((a, "Control (A)", "#1f6f8b"), (b, "Treatment (B)", "#d98324")):
        x = np.sort(g.resolution_hours.clip(upper=60))
        ax[0, 0].plot(x, np.arange(1, len(x) + 1) / len(x), label=name, color=col, lw=2)
    ax[0, 0].set(title=f"ECDF of resolution time (median {r['median_control']:.1f}h -> {r['median_treatment']:.1f}h)",
                 xlabel="Hours (clipped at 60)", ylabel="Share of tickets resolved"); ax[0, 0].legend()
    ests = [("Welch (mean)", r["pct_change"], r["pct_low"], r["pct_high"]),
            ("Bootstrap (mean)", boot["point"], boot["ci_low"], boot["ci_high"]),
            ("CUPED", cu["pct_change"], cu["pct_low"], cu["pct_high"])]
    for i, (n_, p_, lo, hi) in enumerate(ests):
        ax[0, 1].errorbar(p_, i, xerr=[[p_ - lo], [hi - p_]], fmt="o", capsize=6, color="#2a6")
    ax[0, 1].set_yticks(range(3)); ax[0, 1].set_yticklabels([e[0] for e in ests]); ax[0, 1].invert_yaxis()
    ax[0, 1].axvline(0, color="k", ls="--"); ax[0, 1].axvline(-10, color="r", ls=":", label="Planned MDE (-10%)")
    ax[0, 1].set(title="Change in mean resolution time (%, 95% CI)", xlabel="% change"); ax[0, 1].legend()
    for i, s in enumerate(segs):
        ax[0, 2].errorbar(s["pct_change"], i, xerr=[[s["pct_change"] - s["pct_low"]], [s["pct_high"] - s["pct_change"]]], fmt="s", capsize=5, color="#c60")
        ax[0, 2].text(s["pct_high"] + 1.5, i, f"p(Holm)={fmt_p(s['p_holm'])}", va="center", fontsize=8)
    ax[0, 2].set_yticks(range(3)); ax[0, 2].set_yticklabels([s["segment"] for s in segs]); ax[0, 2].invert_yaxis()
    ax[0, 2].axvline(0, color="k", ls="--"); ax[0, 2].set(title="Effect by priority (%)", xlabel="% change", xlim=(-45, 25))
    for k, (g, scale, margin, lab, unit) in enumerate(((g_re, 100, REOPEN_MARGIN * 100, "Reopen rate (7d)", "pp"),
                                                       (g_cs, 1, CSAT_MARGIN, "CSAT", "points"))):
        axx = ax[1, k]
        axx.errorbar(g["diff"] * scale, 0, xerr=[[(g["diff"] - g["ci_low"]) * scale], [(g["ci_high"] - g["diff"]) * scale]],
                     fmt="o", capsize=6, color="#2a6" if g["status"] == "PASS" else "#c22")
        axx.axvline(margin, color="r", ls="--", label=f"Margin {margin:+g}"); axx.axvline(0, color="k", ls=":")
        axx.set_yticks([0]); axx.set_yticklabels([lab]); axx.legend()
        axx.set(title=f"Guardrail: {lab} -> {g['status']}", xlabel=unit)
    ax[1, 2].plot(cum.day, cum.pct, marker="o", color="#d98324")
    ax[1, 2].fill_between(cum.day, cum.lo, cum.hi, alpha=0.2, color="#d98324")
    ax[1, 2].axhline(0, color="k", ls="--"); ax[1, 2].set(title="Cumulative effect by day (95% CI)", xlabel="Days since start", ylabel="% change", ylim=(-40, 25))
    fig.suptitle("Day 3 - A/B test analysis", fontsize=15)
    fig.tight_layout(); fig.savefig(CHARTS / "day3_ab_analysis.png", dpi=150); plt.close(fig)
    print("تم حفظ outputs/charts/day3_ab_analysis.png و outputs/results/ab_results.json")


if __name__ == "__main__":
    main()
