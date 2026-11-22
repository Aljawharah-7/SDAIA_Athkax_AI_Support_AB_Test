"""
دوال مساعدة مشتركة بين ملفات الأيام الأربعة (مسارات نسبية + اختبارات إحصائية مكتوبة يدوياً).
"""
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm, t as tdist

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHARTS = ROOT / "outputs" / "charts"
RESULTS = ROOT / "outputs" / "results"
for _p in (DATA, CHARTS, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

Z975 = norm.ppf(0.975)


def welch(ctrl, trt) -> dict:
    """اختبار Welch لفرق متوسطين (المعالجة - المقارنة) مع فترة ثقة 95%."""
    ctrl, trt = np.asarray(ctrl, float), np.asarray(trt, float)
    n1, n2 = len(ctrl), len(trt)
    v1, v2 = ctrl.var(ddof=1), trt.var(ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    dof = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    d = trt.mean() - ctrl.mean()
    tc = tdist.ppf(0.975, dof)
    return {"mean_control": float(ctrl.mean()), "mean_treatment": float(trt.mean()), "diff": float(d),
            "ci_low": float(d - tc * se), "ci_high": float(d + tc * se),
            "p_value": float(2 * (1 - tdist.cdf(abs(d / se), dof))), "n_control": n1, "n_treatment": n2}


def prop_diff(ctrl, trt) -> dict:
    """اختبار z لفرق نسبتين (المعالجة - المقارنة)."""
    ctrl, trt = np.asarray(ctrl, float), np.asarray(trt, float)
    n1, n2 = len(ctrl), len(trt)
    p1, p2 = ctrl.mean(), trt.mean()
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    pp = (ctrl.sum() + trt.sum()) / (n1 + n2)
    z = (p2 - p1) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    d = p2 - p1
    return {"rate_control": float(p1), "rate_treatment": float(p2), "diff": float(d),
            "ci_low": float(d - Z975 * se), "ci_high": float(d + Z975 * se),
            "p_value": float(2 * (1 - norm.cdf(abs(z)))), "n_control": n1, "n_treatment": n2}


def holm(pvals) -> list:
    """تصحيح Holm-Bonferroni للاختبارات المتعددة."""
    pvals = list(pvals)
    m = len(pvals)
    adj, run = np.empty(m), 0.0
    for rank, idx in enumerate(np.argsort(pvals)):
        run = max(run, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, run)
    return adj.tolist()


def cuped(y, x):
    """CUPED: y_adj = y - theta*(x - mean(x)). تعيد (y_adj, theta, نسبة تقليل التباين)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * (x - x.mean())
    return y_adj, float(theta), float(1 - y_adj.var(ddof=1) / y.var(ddof=1))


def ols_cluster(X, y, groups):
    """OLS مع أخطاء معيارية عنقودية (CR1). تعيد (beta, V, درجات الحرية)."""
    X, y, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ y
    u = y - X @ beta
    meat = np.zeros((X.shape[1],) * 2)
    ids = np.unique(groups)
    for g in ids:
        s = X[groups == g].T @ u[groups == g]
        meat += np.outer(s, s)
    G, n, k = len(ids), len(y), np.linalg.matrix_rank(X)
    V = G / (G - 1) * (n - 1) / (n - k) * bread @ meat @ bread
    return beta, V, G - 1


def coef_ci(beta, V, dof, i) -> dict:
    se = float(np.sqrt(V[i, i]))
    tc = tdist.ppf(0.975, dof)
    return {"coef": float(beta[i]), "se": se, "ci_low": float(beta[i] - tc * se),
            "ci_high": float(beta[i] + tc * se),
            "p_value": float(2 * (1 - tdist.cdf(abs(beta[i] / se), dof)))}


def wald_test(beta, V, idx) -> tuple:
    """اختبار والد المشترك لمعاملات متعددة = 0. تعيد (إحصاءة χ²، القيمة p)."""
    b = beta[idx]
    stat = float(b @ np.linalg.pinv(V[np.ix_(idx, idx)]) @ b)
    return stat, float(1 - chi2.cdf(stat, len(idx)))


def pct(log_effect: float) -> float:
    """تحويل أثر على مقياس اللوغاريتم إلى نسبة تغير مئوية."""
    return float((np.exp(log_effect) - 1) * 100)


def fmt_p(p: float) -> str:
    """تنسيق القيمة p للرسوم."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def bootstrap_pct_change(ctrl, trt, n_boot: int = 2000, seed: int = 5) -> dict:
    """Bootstrap لنسبة التغير في المتوسط: mean(trt)/mean(ctrl) - 1 (بالمئة)."""
    rng = np.random.default_rng(seed)
    ctrl, trt = np.asarray(ctrl, float), np.asarray(trt, float)
    boots = np.array([rng.choice(trt, len(trt)).mean() / rng.choice(ctrl, len(ctrl)).mean() - 1
                      for _ in range(n_boot)]) * 100
    return {"point": float((trt.mean() / ctrl.mean() - 1) * 100),
            "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5))}
