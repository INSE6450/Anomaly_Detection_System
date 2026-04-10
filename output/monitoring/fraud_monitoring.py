import os
import json
import sys
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


@dataclass
class MonitorConfig:
    psi_warn: float = 0.20
    psi_crit: float = 0.30
    js_warn: float = 0.10
    js_crit: float = 0.20
    null_warn: float = 0.01
    null_crit: float = 0.05
    latency_p90_slo_ms: float = 120.0
    error_rate_crit: float = 0.02
    bins: int = 20


def _safe_hist(x, bins=20, eps=1e-8):
    h, _ = np.histogram(x, bins=bins, density=True)
    h = h + eps
    return h / h.sum()


def js_divergence(x_ref, x_cur, bins=20):
    p = _safe_hist(x_ref, bins=bins)
    q = _safe_hist(x_cur, bins=bins)
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


def psi(expected, actual, bins=10, eps=1e-8):
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) < 3:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)
    e = np.clip(e_counts / max(e_counts.sum(), 1), eps, None)
    a = np.clip(a_counts / max(a_counts.sum(), 1), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def compute_data_quality(reference_df, recent_df, key_features, cfg: MonitorConfig):
    rows = []
    missing_cols = [c for c in reference_df.columns if c not in recent_df.columns]
    extra_cols = [c for c in recent_df.columns if c not in reference_df.columns]

    schema_status = "critical" if missing_cols else "ok"
    recent_null_rate = float(recent_df.isnull().mean().mean())
    if recent_null_rate >= cfg.null_crit:
        null_status = "critical"
    elif recent_null_rate >= cfg.null_warn:
        null_status = "warning"
    else:
        null_status = "ok"

    for col in key_features:
        if col not in reference_df.columns or col not in recent_df.columns:
            continue

        x_ref = pd.to_numeric(reference_df[col], errors="coerce").dropna().values
        x_cur = pd.to_numeric(recent_df[col], errors="coerce").dropna().values
        if len(x_ref) < 50 or len(x_cur) < 50:
            continue

        p = psi(x_ref, x_cur, bins=10)
        j = js_divergence(x_ref, x_cur, bins=cfg.bins)
        w = float(wasserstein_distance(x_ref, x_cur))

        status = "ok"
        if p >= cfg.psi_crit or j >= cfg.js_crit:
            status = "critical"
        elif p >= cfg.psi_warn or j >= cfg.js_warn:
            status = "warning"

        rows.append(
            {
                "feature": col,
                "mean_ref": float(np.mean(x_ref)),
                "mean_cur": float(np.mean(x_cur)),
                "std_ref": float(np.std(x_ref)),
                "std_cur": float(np.std(x_cur)),
                "psi": float(p),
                "js": float(j),
                "wasserstein": w,
                "status": status,
                "schema_status": schema_status,
                "null_status": null_status,
                "missing_cols": ",".join(missing_cols) if missing_cols else "",
                "extra_cols": ",".join(extra_cols) if extra_cols else "",
                "recent_null_rate": recent_null_rate,
            }
        )

    return pd.DataFrame(rows)


def compute_model_quality(df, score_col="risk_score", label_col="label", threshold=0.5):
    q = {}
    if score_col in df.columns:
        s = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).values
        q["high_risk_rate"] = float((s >= threshold).mean())
        q["score_mean"] = float(np.mean(s))
    if "complaint" in df.columns:
        q["complaint_rate"] = float(pd.to_numeric(df["complaint"], errors="coerce").fillna(0).mean())
    if "latency_ms" in df.columns:
        lat = pd.to_numeric(df["latency_ms"], errors="coerce").dropna().values
        if len(lat):
            q["latency_p50_ms"] = float(np.percentile(lat, 50))
            q["latency_p90_ms"] = float(np.percentile(lat, 90))
    if "error" in df.columns:
        q["error_rate"] = float(pd.to_numeric(df["error"], errors="coerce").fillna(0).mean())

    if score_col in df.columns and label_col in df.columns:
        y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).values
        s = pd.to_numeric(df[score_col], errors="coerce").fillna(0).values
        yp = (s >= threshold).astype(int)
        if len(np.unique(y)) > 1:
            q["f1"] = float(f1_score(y, yp))
            q["auroc"] = float(roc_auc_score(y, s))
            q["avg_precision"] = float(average_precision_score(y, s))
    return q


def simulate_stream(reference_df, n_batches=12, batch_size=2000, seed=42):
    rng = np.random.default_rng(seed)
    numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Reference dataframe requires numeric columns.")

    start = datetime.now() - timedelta(hours=n_batches)
    batches = []
    for i in range(n_batches):
        sample = reference_df.sample(n=min(batch_size, len(reference_df)), replace=True, random_state=seed + i).copy()

        if i >= n_batches // 2:
            drift_cols = numeric_cols[: min(5, len(numeric_cols))]
            for c in drift_cols:
                sample[c] = sample[c] * 1.08 + 0.15

        base = pd.to_numeric(sample[numeric_cols[0]], errors="coerce").fillna(0.0).values
        risk = 1 / (1 + np.exp(-base))
        risk = np.clip(risk + rng.normal(0, 0.05, len(sample)), 0, 1)

        sample["risk_score"] = risk
        sample["label"] = (risk + rng.normal(0, 0.2, len(sample)) > 0.6).astype(int)
        sample["complaint"] = (risk > 0.8).astype(int) * (rng.random(len(sample)) < 0.15).astype(int)
        sample["latency_ms"] = rng.normal(80 + i * 2, 10, len(sample)).clip(20, 400)
        sample["error"] = (rng.random(len(sample)) < (0.003 + i * 0.0005)).astype(int)
        sample["batch_ts"] = start + timedelta(hours=i)
        sample["batch_id"] = i
        batches.append(sample)

    return pd.concat(batches, ignore_index=True)


def alert_eval(drift_df, quality_df, cfg: MonitorConfig):
    alerts = []
    if not drift_df.empty:
        crit = drift_df[(drift_df["psi"] >= cfg.psi_crit) | (drift_df["js"] >= cfg.js_crit)]
        warn = drift_df[((drift_df["psi"] >= cfg.psi_warn) | (drift_df["js"] >= cfg.js_warn)) & ~drift_df.index.isin(crit.index)]
        if len(crit):
            alerts.append(f"CRITICAL drift: {len(crit)} features")
        if len(warn):
            alerts.append(f"WARNING drift: {len(warn)} features")
        if (drift_df["schema_status"] == "critical").any():
            alerts.append("CRITICAL schema mismatch")
        if (drift_df["null_status"] == "critical").any():
            alerts.append("CRITICAL null-rate spike")

    if "latency_p90_ms" in quality_df.columns:
        if (quality_df["latency_p90_ms"] > cfg.latency_p90_slo_ms).any():
            alerts.append("WARNING latency p90 SLO breach")
    if "error_rate" in quality_df.columns:
        if (quality_df["error_rate"] > cfg.error_rate_crit).any():
            alerts.append("CRITICAL error-rate spike")
    return alerts


def render_dashboard(drift_df, quality_df, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    if not drift_df.empty:
        top_psi = drift_df.sort_values("psi", ascending=False).head(10)
        axes[0, 0].barh(top_psi["feature"], top_psi["psi"])
        axes[0, 0].invert_yaxis()
        axes[0, 0].set_title("Top feature PSI")

        top_js = drift_df.sort_values("js", ascending=False).head(10)
        axes[0, 1].barh(top_js["feature"], top_js["js"], color="orange")
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_title("Top feature JS")
    else:
        axes[0, 0].set_title("No drift data")
        axes[0, 1].set_title("No drift data")

    if not quality_df.empty and "batch_id" in quality_df.columns:
        for c in ["high_risk_rate", "complaint_rate", "error_rate"]:
            if c in quality_df.columns:
                axes[1, 0].plot(quality_df["batch_id"], quality_df[c], marker="o", label=c)
        axes[1, 0].set_title("Proxy metrics")
        axes[1, 0].legend()

        for c in ["latency_p50_ms", "latency_p90_ms"]:
            if c in quality_df.columns:
                axes[1, 1].plot(quality_df["batch_id"], quality_df[c], marker="o", label=c)
        axes[1, 1].set_title("Latency")
        axes[1, 1].legend()
    else:
        axes[1, 0].set_title("No quality data")
        axes[1, 1].set_title("No quality data")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_monitor(reference_path, recent_path, outdir):
    cfg = MonitorConfig()
    os.makedirs(outdir, exist_ok=True)

    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    reference_df = pd.read_csv(reference_path)
    if recent_path and os.path.exists(recent_path):
        recent_df = pd.read_csv(recent_path)
    else:
        recent_df = simulate_stream(reference_df, n_batches=12, batch_size=2000, seed=42)

    ref_num = reference_df.select_dtypes(include=[np.number]).columns.tolist()
    cur_num = recent_df.select_dtypes(include=[np.number]).columns.tolist()
    ignore = {"risk_score", "label", "complaint", "latency_ms", "error", "batch_id"}
    key_features = [c for c in ref_num if c in cur_num and c not in ignore][:20]

    latest = recent_df[recent_df["batch_id"] == recent_df["batch_id"].max()].copy() if "batch_id" in recent_df.columns else recent_df.copy()
    drift_df = compute_data_quality(reference_df, latest, key_features, cfg)

    quality_rows = []
    if "batch_id" in recent_df.columns:
        for bid, grp in recent_df.groupby("batch_id"):
            q = compute_model_quality(grp)
            q["batch_id"] = int(bid)
            quality_rows.append(q)
    else:
        q = compute_model_quality(recent_df)
        q["batch_id"] = 0
        quality_rows.append(q)

    quality_df = pd.DataFrame(quality_rows).sort_values("batch_id")
    alerts = alert_eval(drift_df, quality_df, cfg)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    drift_csv = os.path.join(outdir, f"drift_{ts}.csv")
    qual_csv = os.path.join(outdir, f"quality_{ts}.csv")
    dash_png = os.path.join(outdir, f"dashboard_{ts}.png")
    summary_json = os.path.join(outdir, f"summary_{ts}.json")

    drift_df.to_csv(drift_csv, index=False)
    quality_df.to_csv(qual_csv, index=False)
    render_dashboard(drift_df, quality_df, dash_png)

    summary = {
        "timestamp": ts,
        "reference": reference_path,
        "recent": recent_path if recent_path else "simulated_from_reference",
        "rows_reference": int(len(reference_df)),
        "rows_recent": int(len(recent_df)),
        "alerts": alerts,
        "artifacts": {"drift_csv": drift_csv, "quality_csv": qual_csv, "dashboard_png": dash_png},
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Monitoring run complete.")
    print(f"- {drift_csv}")
    print(f"- {qual_csv}")
    print(f"- {dash_png}")
    print(f"- {summary_json}")
    if alerts:
        print("Alerts:")
        for a in alerts:
            print(f"  * {a}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, default="output/monitoring_reference.csv")
    parser.add_argument("--recent", type=str, default="")
    parser.add_argument("--outdir", type=str, default="output/monitoring")

    # notebook-safe parsing
    if "ipykernel" in sys.modules:
        args, _ = parser.parse_known_args(argv)
    else:
        args = parser.parse_args(argv)

    run_monitor(args.reference, args.recent, args.outdir)


if __name__ == "__main__":
    main()