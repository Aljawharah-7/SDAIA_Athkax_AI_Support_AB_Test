"""
محاكاة تجربة A/B: المساعد الذكي (AI Bot) كخيار أول قبل تحويل التذكرة للدعم البشري
====================================================================================
- المجموعة A (المقارنة): التحويل المباشر للدعم البشري.
- المجموعة B (المعالجة): التوجيه الأول للمساعد الذكي، فإن لم يحل المشكلة تُصعَّد التذكرة للموظف.

البيانات محاكاة (Simulated) لأغراض التدريب. افتراضات المحاكاة مكتوبة في CONFIG أدناه.
"""
import numpy as np
import pandas as pd
from scipy import stats

# إعداد المتغيرات والمعلمات الأساسية للتجربة (دعم فني - AI Bot)
CONFIG = {
    "seed": 2026,
    "daily_tickets": 250,            # عدد التذاكر الجديدة يومياً
    "treatment_share": 0.5,          # نسبة التذاكر الموجهة للمساعد الذكي (المجموعة B)
    "n_agents": 40,                  # عدد موظفي الدعم
    "categories": ["Account", "Software", "Network", "Hardware"],
    "category_probs": [0.30, 0.30, 0.20, 0.20],
    "human_base_log": {"Account": 1.6, "Software": 2.1, "Network": 2.3, "Hardware": 2.5},  # log(ساعات) عند الموظف
    "priority_shift": {"Low": 0.15, "Medium": 0.0, "High": -0.20},
    "sigma_log": 0.70,               # تباين زمن الحل لدى الموظف
    "complexity_strength": 0.25,     # أثر درجة تعقيد التذكرة على log(الزمن)
    # --- أثر المساعد الذكي (الحقيقة المُدخَلة في المحاكاة) ---
    "bot_resolve_rate": {"Account": 0.18, "Software": 0.12, "Network": 0.04, "Hardware": 0.01},
    "bot_hours_median": 0.4,         # وسيط زمن الحل عندما يحلها المساعد بنفسه (ساعة)
    "bot_attempt_delay_h": 0.15,     # وقت محاولة المساعد قبل التصعيد (ساعة)
    "summary_effect": {"Account": -0.05, "Software": -0.04, "Network": -0.02, "Hardware": 0.01},  # ملخص المساعد يسرّع/يبطئ الموظف
    "novelty_boost": 0.10,           # فضول المستخدمين في الأيام الأولى يرفع نسبة استخدام المساعد
    "first_response_bot_min": 1.0,   # أول رد من المساعد (دقيقة)
    "first_response_human_min": 30.0,
    # --- مقاييس الحماية ---
    "reopen_rate": 0.08, "reopen_hardware_extra": 0.03, "reopen_bot_extra": 0.04,
    "csat_mean": 4.0, "csat_bot_effect": -0.25, "csat_response_rate": 0.40,
    "srm_bug_share": 0.0,            # نسبة تذاكر B المفقودة بسبب خلل تسجيل (للتدريب على اكتشاف SRM)
}


def simulate_experiment(cfg, duration_days=28):
    r = np.random.default_rng(cfg["seed"])
    n = duration_days * cfg["daily_tickets"]
    is_b = (r.random(n) < cfg["treatment_share"]).astype(int)
    day = r.integers(1, duration_days + 1, n)
    agent_id = r.integers(1, cfg["n_agents"] + 1, n)
    skill = r.normal(0, 0.25, cfg["n_agents"] + 1)                       # مهارة الموظف
    agent_pre = np.exp(2.0 + skill + r.normal(0, 0.05, cfg["n_agents"] + 1))   # متوسط ساعات الموظف قبل التجربة

    category = r.choice(cfg["categories"], n, p=cfg["category_probs"])
    priority = r.choice(list(cfg["priority_shift"]), n, p=[0.30, 0.50, 0.20])
    complexity = r.normal(0, 1, n)

    human_log = (np.array([cfg["human_base_log"][c] for c in category])
                 + np.array([cfg["priority_shift"][p] for p in priority])
                 + cfg["complexity_strength"] * complexity + skill[agent_id] + r.normal(0, cfg["sigma_log"], n))
    human_hours = np.exp(human_log)

    novelty = 1 + cfg["novelty_boost"] * np.exp(-(day - 1) / 3)
    p_bot = np.clip(np.array([cfg["bot_resolve_rate"][c] for c in category]) * novelty, 0, 1)
    bot_resolved = is_b * r.binomial(1, p_bot)
    bot_hours = r.lognormal(np.log(cfg["bot_hours_median"]), 0.5, n)
    summary = np.exp(np.array([cfg["summary_effect"][c] for c in category]))
    hours = np.where(bot_resolved == 1, bot_hours,
                     human_hours * np.where(is_b == 1, summary, 1.0) + cfg["bot_attempt_delay_h"] * is_b)
    hours = np.clip(hours, 0.1, None)

    first_resp = np.where(is_b == 1, r.lognormal(np.log(cfg["first_response_bot_min"]), 0.5, n),
                          r.lognormal(np.log(cfg["first_response_human_min"]), 0.6, n))
    p_reopen = (cfg["reopen_rate"] + cfg["reopen_hardware_extra"] * (category == "Hardware")
                + cfg["reopen_bot_extra"] * bot_resolved)
    reopened = r.binomial(1, np.clip(p_reopen, 0, 1))
    csat_full = np.clip(np.round(r.normal(cfg["csat_mean"] + cfg["csat_bot_effect"] * bot_resolved, 0.9)), 1, 5)
    csat = np.where(r.random(n) < cfg["csat_response_rate"], csat_full, np.nan)

    df = pd.DataFrame({
        "ticket_id": np.arange(1, n + 1), "variant": np.where(is_b == 1, "B", "A"), "day": day,
        "agent_id": agent_id, "category": category, "priority": priority,
        "complexity": complexity.round(3), "agent_pre_avg_hours": agent_pre[agent_id].round(3),
        "resolution_hours": hours.round(3), "first_response_min": first_resp.round(2),
        "bot_resolved": bot_resolved, "escalated_to_human": 1 - bot_resolved,
        "reopened_7d": reopened, "csat": csat})
    if cfg["srm_bug_share"] > 0:                                          # محاكاة خلل يفقد جزءاً من تذاكر B
        lost = (is_b == 1) & (r.random(n) < cfg["srm_bug_share"])
        df = df[~lost]
    return df.reset_index(drop=True)


# فحص SRM (Sample Ratio Mismatch)
def srm_check(obs_a, obs_b, expected_share=0.5, verbose=True):
    total = obs_a + obs_b
    exp_a = total * (1 - expected_share)
    exp_b = total * expected_share
    chi2 = ((obs_a - exp_a) ** 2 / exp_a) + ((obs_b - exp_b) ** 2 / exp_b)
    p_val = 1 - stats.chi2.cdf(chi2, df=1)
    if verbose:
        print(f"فحص SRM - مجموعة A: {obs_a}, مجموعة B: {obs_b} | قيمة p-value لـ SRM: {p_val:.4f}")
    return {"chi2": float(chi2), "p_value": float(p_val), "status": "PASS" if p_val > 0.001 else "FAIL (SRM)"}
