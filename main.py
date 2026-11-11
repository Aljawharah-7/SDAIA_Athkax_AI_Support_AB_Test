import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# التأكد من وجود المجلدات المطلوبة لتجنب أي أخطاء
os.makedirs("data", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

print("--- بدء محاكاة وتجهيز بيانات مشروع A/B Testing ---")

# إعداد المتغيرات والمعلمات الأساسية للتجربة (دعم فني - AI Bot)
CONFIG = {
    "seed": 2026,
    "daily_new_users": 2000,
    "treatment_share": 0.5,
    "baseline_conversion": 0.3,
    "activity_strength": 0.5,
    "true_lift_abs": 0.03,
    "novelty_boost": 0.1,
    "order_value_mean": 150,
    "order_value_sd": 30,
    "order_value_effect": 5,
    "margin": 0.2,
    "extra_cost_per_order_b": 0.5,
    "cancel_rate": 0.05,
    "cancel_effect": -0.01,
    "support_rate": 0.4,
    "support_effect": -0.15, # أثر المساعد الذكي في تقليل وقت/حاجة التذاكر
    "srm_bug_share": 0.0
}

def simulate_experiment(cfg, duration_days=30):
    r = np.random.default_rng(cfg["seed"])
    n = duration_days * cfg["daily_new_users"]
    is_b = (r.random(n) < cfg["treatment_share"]).astype(int)
    day = r.integers(1, duration_days + 1, n)
    activity = r.normal(0, 1, n)
    base = cfg["baseline_conversion"]
    logit = np.log(base / (1 - base)) + cfg["activity_strength"] * activity
    p0 = 1 / (1 + np.exp(-logit))
    p0 = np.clip(p0, 0, 1)
    pre_orders_90d = r.poisson(5 * p0 / base)
    
    lift = cfg["true_lift_abs"] * (1 + cfg["novelty_boost"] * np.exp(-(day - 1) / 3))
    ordered = r.binomial(1, np.clip(p0 + lift * is_b, 0, 1))
    
    value = np.maximum(10, r.normal(cfg["order_value_mean"] + cfg["order_value_effect"] * is_b, cfg["order_value_sd"]))
    order_value = np.where(ordered == 1, value, np.nan)
    revenue = ordered * value
    profit = ordered * (cfg["margin"] * value - cfg["extra_cost_per_order_b"] * is_b)
    cancelled = np.where(ordered == 1, r.binomial(1, np.clip(cfg["cancel_rate"] + cfg["cancel_effect"] * is_b, 0, 1)), np.nan)
    support = r.binomial(1, np.clip(cfg["support_rate"] + cfg["support_effect"] * is_b, 0, 1))
    
    df = pd.DataFrame({
        "variant": np.where(is_b == 1, "B", "A"),
        "day": day,
        "pre_orders_90d": pre_orders_90d,
        "ordered": ordered,
        "order_value": order_value,
        "revenue": revenue,
        "profit": profit,
        "cancelled": cancelled,
        "contacted_support": support
    })
    return df.reset_index(drop=True)

# توليد البيانات وحفظها في مجلد data تلقائياً
data = simulate_experiment(CONFIG, duration_days=30)
data.to_csv("data/dataset.csv", index=False)
print("تم توليد البيانات بنجاح وحفظها في: data/dataset.csv")
print(f"عدد السجلات الإجمالي: {len(data)}")

# فحص الـ SRM (Sample Ratio Mismatch)
def srm_check(obs_a, obs_b, expected_share=0.5):
    total = obs_a + obs_b
    exp_a = total * (1 - expected_share)
    exp_b = total * expected_share
    chi2 = ((obs_a - exp_a)**2 / exp_a) + ((obs_b - exp_b)**2 / exp_b)
    from scipy import stats
    p_val = 1 - stats.chi2.cdf(chi2, df=1)
    print(f"فحص SRM - مجموعة A: {obs_a}, مجموعة B: {obs_b} | قيمة p-value لـ SRM: {p_val:.4f}")

counts = data["variant"].value_counts()
srm_check(counts.get("A", 0), counts.get("B", 0), CONFIG["treatment_share"])

print("--- انتهى تشخيص وتجهيز البيانات بنجاح ---")