"""
اليوم الأول: السؤال والفرضية وتصميم التجربة وحجم العينة
=============================================================
المشروع : تقييم أثر المساعد الذكي (AI Bot) على زمن حل التذاكر في الدعم الفني
الدورة  : التجارب واختبارات A/B والاستدلال السببي (أكاديمية سدايا)

الفكرة المحورية: حجم العينة المعتمد = الأكبر بين متطلبات المقياس الرئيسي ومقاييس الحماية،
حتى لا تنتهي التجربة بمقياس حماية "غير محسوم" بسبب ضعف القوة الإحصائية.
"""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from utils import CHARTS, RESULTS

ALPHA, POWER = 0.05, 0.80
BASELINE_MEAN_H = 11.5      # متوسط زمن الحل الحالي (ساعات) من بيانات تاريخية افتراضية
SD_H = 11.6                 # الانحراف المعياري لزمن الحل (توزيع منحرف بذيل طويل)
MDE_RELATIVE = 0.10         # أصغر أثر مهم تجارياً: تقليل 10% في متوسط الزمن
REOPEN_BASE, REOPEN_MARGIN = 0.085, 0.02   # إعادة الفتح خلال 7 أيام: الأساس 8.5% والهامش +2 نقطة
TICKETS_PER_DAY = 250


def n_mean(delta_abs: float, sd: float, alpha=ALPHA, power=POWER) -> int:
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil(2 * (z * sd / delta_abs) ** 2)


def n_noninferiority(p: float, margin: float, alpha=ALPHA, power=POWER) -> int:
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil(2 * p * (1 - p) * (z / margin) ** 2)


def main() -> None:
    delta = BASELINE_MEAN_H * MDE_RELATIVE
    n_primary = n_mean(delta, SD_H)
    n_guard = n_noninferiority(REOPEN_BASE, REOPEN_MARGIN)
    n_final = max(n_primary, n_guard)
    total = 2 * n_final
    days_raw = math.ceil(total / TICKETS_PER_DAY)
    days = math.ceil(days_raw / 7) * 7

    design = {
        "question": "هل يؤدي اعتماد المساعد الذكي كخيار أول إلى تقليل الوقت المستغرق لحل التذاكر؟",
        "hypothesis": "تقديم المساعد الذكي يقلل زمن الحل من خلال التعامل الفوري مع الاستفسارات الشائعة وتقليل الضغط على ممثلي الدعم البشري.",
        "treatment_group": "B: التوجيه الأول للمساعد الذكي ثم التصعيد للموظف عند الحاجة",
        "control_group": "A: التحويل المباشر للدعم البشري",
        "randomization_unit": "التذكرة (عشوائي 50/50) مع أخطاء معيارية عنقودية حسب الموظف",
        "primary_metric": "متوسط زمن حل التذكرة (ساعات)",
        "secondary_metric": "زمن أول رد (دقائق)",
        "guardrails": {"reopened_7d": f"نسبة إعادة فتح/تكرار التذكرة خلال 7 أيام (هامش +{REOPEN_MARGIN*100:.0f} نقطة)",
                       "csat": "رضا العملاء 1-5 (هامش -0.20)"},
        "alpha": ALPHA, "power": POWER, "baseline_mean_hours": BASELINE_MEAN_H, "sd_hours": SD_H,
        "mde_relative": MDE_RELATIVE, "mde_hours": delta,
        "n_primary_per_group": n_primary, "n_guardrail_per_group": n_guard, "n_per_group": n_final,
        "total_tickets": total, "tickets_per_day": TICKETS_PER_DAY, "days_raw": days_raw, "planned_days": days,
        "assumptions": ["تقريب طبيعي (حجم العينة كبير فيصح مع التوزيع المنحرف)", "توزيع 50/50",
                        "الانحراف المعياري لزمن الحل 11.6 ساعة", "نسبة إعادة الفتح الأساسية 8.5%"],
    }
    (RESULTS / "design.json").write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"حجم العينة: رئيسي {n_primary:,} | حماية {n_guard:,} -> المعتمد {n_final:,} لكل مجموعة")
    print(f"الإجمالي {total:,} تذكرة | {days_raw} يوماً -> المدة المخططة {days} يوماً")

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ns = np.arange(200, 6001, 100)
    for mde in (0.06, 0.08, 0.10, 0.12):
        d = BASELINE_MEAN_H * mde
        ax[0, 0].plot(ns, norm.cdf(d / (SD_H * np.sqrt(2 / ns)) - norm.ppf(0.975)), label=f"MDE {mde:.0%}")
    ax[0, 0].axhline(POWER, color="gray", ls="--"); ax[0, 0].axvline(n_primary, color="r", ls=":")
    ax[0, 0].set(title="Primary metric: power vs tickets per group", xlabel="Tickets per group", ylabel="Power")
    ax[0, 0].legend()
    ax[0, 1].bar(["Primary\n(mean resolution time)", "Guardrail\n(reopen rate)"], [n_primary, n_guard], color=["#1f6f8b", "#d98324"])
    for i, v in enumerate([n_primary, n_guard]):
        ax[0, 1].text(i, v + 40, f"{v:,}", ha="center", fontweight="bold")
    ax[0, 1].set(title="Required tickets per group (largest wins)", ylabel="Tickets per group")
    margins = np.arange(0.01, 0.051, 0.0025)
    ax[1, 0].plot(margins * 100, [n_noninferiority(REOPEN_BASE, m) for m in margins], color="#a33", lw=2)
    ax[1, 0].scatter([REOPEN_MARGIN * 100], [n_guard], color="k", zorder=5)
    ax[1, 0].set(title="Guardrail sample size vs margin", xlabel="Margin (pp)", ylabel="Tickets per group")
    tr = np.arange(100, 801, 25)
    ax[1, 1].plot(tr, [math.ceil(total / t) for t in tr], color="#4c9a2a", lw=2)
    ax[1, 1].axvline(TICKETS_PER_DAY, color="r", ls=":")
    ax[1, 1].set(title="Days needed vs daily ticket volume", xlabel="Tickets per day", ylabel="Days")
    fig.suptitle("Day 1 - Experiment design: AI Bot as first-line support", fontsize=14)
    fig.tight_layout(); fig.savefig(CHARTS / "day1_design.png", dpi=150); plt.close(fig)
    print("تم حفظ outputs/charts/day1_design.png")


if __name__ == "__main__":
    main()
