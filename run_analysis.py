#!/usr/bin/env python3
"""
Semiconductor Manufacturing Anomaly Detection using PCA-based Methods
Data: LAM 9600 Metal Etcher (MACHINE_Data.mat)
Methods: PCA (T² & SPE), Isolation Forest, One-Class SVM
"""

import os
import numpy as np
import pandas as pd
import scipy.io
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import warnings

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
DATA_FILE = "MACHINE_Data.mat"
PLOTS_DIR = "plots"
RESULTS_DIR = "results"
ALPHA = 0.95  # confidence level for control limits
DPI = 300

np.random.seed(42)

# ── Helper functions ───────────────────────────────────────────────────────


def create_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    """Load and parse the LAM 9600 etcher data. Handles variable-length wafers."""
    mat = scipy.io.loadmat(DATA_FILE, squeeze_me=True, struct_as_record=False)
    ld = mat["LAMDATA"]

    calib_names = [str(n).strip() for n in ld.calib_names]
    test_names = [str(n).strip() for n in ld.test_names]
    fault_names = [str(n).strip() for n in ld.fault_names]
    variables = [str(v).strip() for v in ld.variables]

    # Extract calibration wafers as list of 2D arrays (handle different row counts)
    calib_list = [np.atleast_2d(ld.calibration[i]) for i in range(len(ld.calibration))]
    test_list = [np.atleast_2d(ld.test[i]) for i in range(len(ld.test))]

    # Filter out extremely short calibration wafers (< 30 rows) — likely incomplete
    MIN_ROWS = 30
    keep_calib = [i for i, w in enumerate(calib_list) if w.shape[0] >= MIN_ROWS]
    excluded = [i for i, w in enumerate(calib_list) if w.shape[0] < MIN_ROWS]
    if excluded:
        print(
            f"  Excluding {len(excluded)} calibration wafer(s) with < {MIN_ROWS} rows: "
            f"{[calib_names[i] for i in excluded]}"
        )

    calib_list = [calib_list[i] for i in keep_calib]
    calib_names = [calib_names[i] for i in keep_calib]

    return calib_list, test_list, calib_names, test_names, fault_names, variables


def preprocess(calib_list, test_list):
    """
    Stack all time-step observations as rows.
    Trim all wafers to common row count (minimum across all).
    Normalize to zero mean, unit variance using calibration statistics.
    """
    n_calib = len(calib_list)
    n_test = len(test_list)
    n_vars = calib_list[0].shape[1]

    # Find common minimum rows across all wafers
    all_rows = [w.shape[0] for w in calib_list] + [w.shape[0] for w in test_list]
    min_rows = min(all_rows)
    print(f"  Common time rows (trimming all wafers to): {min_rows}")

    # Trim and stack calibration
    X_calib = np.vstack([w[:min_rows, :] for w in calib_list])
    # Trim and stack test
    X_test = np.vstack([w[:min_rows, :] for w in test_list])

    # Fit scaler on calibration data
    scaler = StandardScaler()
    X_calib_scaled = scaler.fit_transform(X_calib)
    X_test_scaled = scaler.transform(X_test)

    return X_calib_scaled, X_test_scaled, scaler, min_rows, n_calib, n_test


def perform_pca(X_calib, variables):
    """
    PCA on calibration data.
    Determine components using Kaiser criterion (eigenvalue > 1) and scree plot.
    """
    # Full PCA to get all eigenvalues
    pca_full = PCA()
    scores_full = pca_full.fit_transform(X_calib)

    eigenvalues = pca_full.explained_variance_
    explained_ratio = pca_full.explained_variance_ratio_
    cumulative_ratio = np.cumsum(explained_ratio)

    # Kaiser criterion: eigenvalue > 1 (on correlation matrix, so eigenvalues are relative to std)
    kaiser_components = np.sum(eigenvalues > 1.0)

    # Scree plot analysis: find elbow point
    # Use the point where cumulative variance explains >= 90%
    n_90 = np.searchsorted(cumulative_ratio, 0.90) + 1

    print(f"All eigenvalues: {eigenvalues[:10].round(3)}")
    print(f"Kaiser criterion (eigenvalue > 1): {kaiser_components} components")
    print(f"90% variance threshold: {n_90} components")
    print(
        f"Variance explained by first {kaiser_components} PCs: {cumulative_ratio[kaiser_components - 1] * 100:.1f}%"
    )

    # Use Kaiser criterion as primary
    n_components = kaiser_components

    # Retrain PCA with selected components
    pca = PCA(n_components=n_components)
    scores_calib = pca.fit_transform(X_calib)

    # Compute calibration statistics
    loadings = pca.components_  # (n_components, n_vars)

    # T² statistics for calibration
    T2_calib = np.sum((scores_calib / np.sqrt(pca.explained_variance_)) ** 2, axis=1)

    # SPE for calibration (residual sum of squares)
    X_reconstructed = pca.inverse_transform(scores_calib)
    SPE_calib = np.sum((X_calib - X_reconstructed) ** 2, axis=1)

    return (
        pca,
        scores_calib,
        eigenvalues,
        explained_ratio,
        cumulative_ratio,
        kaiser_components,
        T2_calib,
        SPE_calib,
        loadings,
    )


def compute_T2_limit(n_components, n_samples, alpha=0.95):
    """
    T²_limit = [r*(m-1)/(m-r)] * F(r, m-r; alpha)
    where r = n_components, m = n_samples
    """
    r = n_components
    m = n_samples
    # Use the first test batch size as m for practical limit
    scaling = r * (m - 1) / (m - r)
    f_val = stats.f.ppf(alpha, r, m - r)
    T2_limit = scaling * f_val
    return T2_limit


def compute_SPE_limit(SPE_calib, alpha=0.95):
    """
    Chi-square approximation for SPE limit:
    SPE_limit = g * chi2(h; alpha)
    where g = SPE_var / (2 * SPE_mean)
          h = 2 * SPE_mean^2 / SPE_var
    """
    m = np.mean(SPE_calib)
    v = np.var(SPE_calib)
    g = v / (2 * m)
    h = 2 * m**2 / v
    SPE_limit = g * stats.chi2.ppf(alpha, h)
    return SPE_limit, g, h


def compute_T2_SPE_test(X_test, pca, T2_limit, SPE_limit):
    """Compute T² and SPE for test data."""
    scores_test = pca.transform(X_test)
    eigenvalues = pca.explained_variance_

    T2_test = np.sum((scores_test / np.sqrt(eigenvalues)) ** 2, axis=1)

    X_reconstructed = pca.inverse_transform(scores_test)
    SPE_test = np.sum((X_test - X_reconstructed) ** 2, axis=1)

    return T2_test, SPE_test, scores_test


def compute_contributions(
    X_test, pca, n_samples_per_fault, n_faults, n_rows, variables
):
    """
    Compute variable contributions to T² and SPE per fault type.
    Returns average contribution per variable per fault.
    """
    n_vars = len(variables)
    loadings = pca.components_  # (r, n_vars)
    eigenvalues = pca.explained_variance_
    r = pca.n_components_

    T2_contrib_all = np.zeros((n_faults, n_vars))
    SPE_contrib_all = np.zeros((n_faults, n_vars))

    for f in range(n_faults):
        start = f * n_rows
        end = (f + 1) * n_rows
        X_f = X_test[start:end]
        scores_f = pca.transform(X_f)
        X_recon = pca.inverse_transform(scores_f)
        residuals = X_f - X_recon

        # T² contribution: for each variable j, sum over components k:
        # |p_jk * t_k / sqrt(lambda_k)| (proportional contribution)
        T2_contrib = np.zeros((n_rows, n_vars))
        for k in range(r):
            for j in range(n_vars):
                T2_contrib[:, j] += np.abs(
                    loadings[k, j] * scores_f[:, k] / np.sqrt(eigenvalues[k])
                )

        # SPE contribution: residual squared per variable
        SPE_contrib = residuals**2

        T2_contrib_all[f] = np.mean(T2_contrib, axis=0)
        SPE_contrib_all[f] = np.mean(SPE_contrib, axis=0)

    return T2_contrib_all, SPE_contrib_all


def run_benchmarks(X_calib, X_test, n_rows, n_faults, fault_names):
    """Run Isolation Forest and One-Class SVM benchmarks."""
    # Fit on calibration data
    iso_forest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso_forest.fit(X_calib)

    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(X_calib)

    # Predict on test data
    iso_pred = iso_forest.predict(X_test)  # 1 = inlier, -1 = outlier
    svm_pred = ocsvm.predict(X_test)

    # Detection rate per fault: fraction of time samples flagged as anomaly
    iso_rates = []
    svm_rates = []
    for f in range(n_faults):
        start = f * n_rows
        end = (f + 1) * n_rows
        iso_f = iso_pred[start:end]
        svm_f = svm_pred[start:end]
        iso_rates.append(np.mean(iso_f == -1))
        svm_rates.append(np.mean(svm_f == -1))

    return iso_pred, svm_pred, iso_rates, svm_rates


def compute_detection_rates(T2_test, SPE_test, T2_limit, SPE_limit, n_rows, n_faults):
    """Compute detection rate per fault type for T² and SPE."""
    T2_rates = []
    SPE_rates = []
    for f in range(n_faults):
        start = f * n_rows
        end = (f + 1) * n_rows
        T2_f = T2_test[start:end]
        SPE_f = SPE_test[start:end]
        T2_rates.append(np.mean(T2_f > T2_limit))
        SPE_rates.append(np.mean(SPE_f > SPE_limit))
    return T2_rates, SPE_rates


# ── Plotting functions ─────────────────────────────────────────────────────


def plot_scree(eigenvalues, explained_ratio, cumulative_ratio, kaiser_n):
    """Scree plot with Kaiser criterion marker."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    n_show = min(20, len(eigenvalues))
    x = np.arange(1, n_show + 1)

    # Eigenvalue plot
    ax1.plot(x, eigenvalues[:n_show], "bo-", markersize=6, linewidth=1.5)
    ax1.axhline(
        y=1.0,
        color="r",
        linestyle="--",
        linewidth=1.2,
        label="Kaiser criterion (eigenvalue = 1)",
    )
    ax1.axvline(
        x=kaiser_n,
        color="g",
        linestyle=":",
        linewidth=1.2,
        label=f"Selected: {kaiser_n} components",
    )
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Eigenvalue", fontsize=12)
    ax1.set_title("Scree Plot (Eigenvalues)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3)

    # Cumulative variance
    ax2.plot(x, cumulative_ratio[:n_show] * 100, "rs-", markersize=6, linewidth=1.5)
    ax2.axhline(
        y=90, color="orange", linestyle="--", linewidth=1.2, label="90% threshold"
    )
    ax2.axvline(
        x=kaiser_n,
        color="g",
        linestyle=":",
        linewidth=1.2,
        label=f"Selected: {kaiser_n} PCs ({cumulative_ratio[kaiser_n - 1] * 100:.1f}%)",
    )
    ax2.set_xlabel("Number of Principal Components", fontsize=12)
    ax2.set_ylabel("Cumulative Variance Explained (%)", fontsize=12)
    ax2.set_title("Cumulative Variance Explained", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_xticks(x)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "scree_plot.png"), dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_T2_SPE_charts(
    T2_calib,
    SPE_calib,
    T2_test,
    SPE_test,
    T2_limit,
    SPE_limit,
    n_rows,
    n_faults,
    fault_names,
):
    """T² and SPE control charts for calibration and all test faults."""
    # Combined figure: 2 columns (T², SPE), 1+ n_faults rows
    n_rows_fig = 1 + n_faults  # calibration + each fault
    fig, axes = plt.subplots(n_rows_fig, 2, figsize=(16, 4 * n_rows_fig))

    # Calibration row
    ax_t2, ax_spe = axes[0]
    samples_calib = np.arange(len(T2_calib))
    ax_t2.plot(samples_calib, T2_calib, "b.", markersize=1, alpha=0.5)
    ax_t2.axhline(
        T2_limit, color="r", linewidth=1.5, label=f"T² limit = {T2_limit:.1f}"
    )
    ax_t2.set_title("Calibration Data - T²", fontsize=11)
    ax_t2.set_ylabel("T²")
    ax_t2.legend(fontsize=9)
    ax_t2.grid(True, alpha=0.3)

    ax_spe.plot(samples_calib, SPE_calib, "b.", markersize=1, alpha=0.5)
    ax_spe.axhline(
        SPE_limit, color="r", linewidth=1.5, label=f"SPE limit = {SPE_limit:.1f}"
    )
    ax_spe.set_title("Calibration Data - SPE", fontsize=11)
    ax_spe.set_ylabel("SPE")
    ax_spe.legend(fontsize=9)
    ax_spe.grid(True, alpha=0.3)

    # Fault rows
    for f in range(n_faults):
        start = f * n_rows
        end = (f + 1) * n_rows
        samples = np.arange(n_rows)

        T2_f = T2_test[start:end]
        SPE_f = SPE_test[start:end]

        ax_t2, ax_spe = axes[f + 1]

        ax_t2.plot(samples, T2_f, "b-", linewidth=0.8, alpha=0.7)
        ax_t2.axhline(
            T2_limit, color="r", linewidth=1.2, label=f"Limit = {T2_limit:.1f}"
        )
        above = np.where(T2_f > T2_limit)[0]
        if len(above) > 0:
            ax_t2.plot(
                above,
                T2_f[above],
                "rx",
                markersize=5,
                label=f"Exceed ({len(above)}/{n_rows})",
            )
        rate = np.mean(T2_f > T2_limit) * 100
        ax_t2.set_title(f"{fault_names[f]} - T² (det: {rate:.0f}%)", fontsize=10)
        ax_t2.set_ylabel("T²")
        ax_t2.legend(fontsize=8, loc="upper right")
        ax_t2.grid(True, alpha=0.3)

        ax_spe.plot(samples, SPE_f, "b-", linewidth=0.8, alpha=0.7)
        ax_spe.axhline(
            SPE_limit, color="r", linewidth=1.2, label=f"Limit = {SPE_limit:.1f}"
        )
        above = np.where(SPE_f > SPE_limit)[0]
        if len(above) > 0:
            ax_spe.plot(
                above,
                SPE_f[above],
                "rx",
                markersize=5,
                label=f"Exceed ({len(above)}/{n_rows})",
            )
        rate = np.mean(SPE_f > SPE_limit) * 100
        ax_spe.set_title(f"{fault_names[f]} - SPE (det: {rate:.0f}%)", fontsize=10)
        ax_spe.set_ylabel("SPE")
        ax_spe.legend(fontsize=8, loc="upper right")
        ax_spe.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time Step", fontsize=11)
    axes[-1, 1].set_xlabel("Time Step", fontsize=11)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, "T2_SPE_control_charts.png"),
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close()


def plot_individual_fault_charts(
    T2_test, SPE_test, T2_limit, SPE_limit, n_rows, fault_names
):
    """Individual T² and SPE charts per fault type."""
    for f, fname in enumerate(fault_names):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        start = f * n_rows
        end = (f + 1) * n_rows
        samples = np.arange(n_rows)

        T2_f = T2_test[start:end]
        SPE_f = SPE_test[start:end]

        ax1.plot(samples, T2_f, "b-", linewidth=0.8)
        ax1.axhline(
            T2_limit, color="r", linewidth=1.5, label=f"T² limit = {T2_limit:.1f}"
        )
        above = np.where(T2_f > T2_limit)[0]
        if len(above) > 0:
            ax1.plot(above, T2_f[above], "rx", markersize=6)
        ax1.set_title(f"{fname} - Hotelling T²", fontsize=13)
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("T²")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(samples, SPE_f, "b-", linewidth=0.8)
        ax2.axhline(
            SPE_limit, color="r", linewidth=1.5, label=f"SPE limit = {SPE_limit:.1f}"
        )
        above = np.where(SPE_f > SPE_limit)[0]
        if len(above) > 0:
            ax2.plot(above, SPE_f[above], "rx", markersize=6)
        ax2.set_title(f"{fname} - SPE (Squared Prediction Error)", fontsize=13)
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("SPE")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_name = fname.strip().replace(" ", "_").replace("+", "p").replace("-", "m")
        plt.savefig(
            os.path.join(PLOTS_DIR, f"chart_{f:02d}_{safe_name}.png"),
            dpi=DPI,
            bbox_inches="tight",
        )
        plt.close()


def plot_contribution_bars(T2_contrib, SPE_contrib, fault_names, variables):
    """Contribution bar plots for each fault type."""
    for f, fname in enumerate(fault_names):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # T² contributions
        colors_t2 = plt.cm.Reds(T2_contrib[f] / T2_contrib[f].max() * 0.8 + 0.1)
        bars = ax1.barh(variables, T2_contrib[f], color=colors_t2)
        ax1.set_xlabel("Average T² Contribution", fontsize=11)
        ax1.set_title(f"{fname} - T² Variable Contributions", fontsize=12)
        ax1.grid(True, alpha=0.3, axis="x")

        # SPE contributions
        colors_spe = plt.cm.Blues(
            SPE_contrib[f] / max(SPE_contrib[f].max(), 1e-10) * 0.8 + 0.1
        )
        ax2.barh(variables, SPE_contrib[f], color=colors_spe)
        ax2.set_xlabel("Average SPE Contribution", fontsize=11)
        ax2.set_title(f"{fname} - SPE Variable Contributions", fontsize=12)
        ax2.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        safe_name = fname.strip().replace(" ", "_").replace("+", "p").replace("-", "m")
        plt.savefig(
            os.path.join(PLOTS_DIR, f"contrib_{f:02d}_{safe_name}.png"),
            dpi=DPI,
            bbox_inches="tight",
        )
        plt.close()


def plot_sensitivity_analysis(X_calib, variables):
    """Sensitivity analysis: effect of number of components on detection."""
    eigenvalues_full = np.linalg.eigvalsh(np.corrcoef(X_calib.T))[::-1]
    max_k = min(20, len(eigenvalues_full))
    ks = np.arange(1, max_k + 1)

    cum_var = []
    for k in ks:
        pca_k = PCA(n_components=k)
        pca_k.fit(X_calib)
        cum_var.append(np.sum(pca_k.explained_variance_ratio_) * 100)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, cum_var, "bo-", markersize=6, linewidth=1.5)
    kaiser_n = np.sum(eigenvalues_full > 1.0)
    ax.axvline(
        kaiser_n,
        color="g",
        linestyle=":",
        linewidth=1.5,
        label=f"Kaiser: {kaiser_n} PCs",
    )
    ax.axhline(90, color="orange", linestyle="--", linewidth=1.2, label="90% threshold")
    ax.set_xlabel("Number of Principal Components", fontsize=12)
    ax.set_ylabel("Cumulative Variance Explained (%)", fontsize=12)
    ax.set_title("PCA Component Selection - Sensitivity Analysis", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, "sensitivity_analysis.png"),
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close()


def plot_detection_comparison(fault_names, T2_rates, SPE_rates, iso_rates, svm_rates):
    """Grouped bar chart comparing detection rates."""
    n_faults = len(fault_names)
    x = np.arange(n_faults)
    width = 0.2

    fig, ax = plt.subplots(figsize=(16, 7))
    bars1 = ax.bar(
        x - 1.5 * width, [r * 100 for r in T2_rates], width, label="T²", color="#2196F3"
    )
    bars2 = ax.bar(
        x - 0.5 * width,
        [r * 100 for r in SPE_rates],
        width,
        label="SPE",
        color="#FF5722",
    )
    bars3 = ax.bar(
        x + 0.5 * width,
        [r * 100 for r in iso_rates],
        width,
        label="Isolation Forest",
        color="#4CAF50",
    )
    bars4 = ax.bar(
        x + 1.5 * width,
        [r * 100 for r in svm_rates],
        width,
        label="One-Class SVM",
        color="#9C27B0",
    )

    ax.set_xlabel("Fault Type", fontsize=12)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_title("Fault Detection Rate Comparison: PCA vs ML Methods", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [n.strip() for n in fault_names], rotation=45, ha="right", fontsize=9
    )
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )

    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, "detection_comparison.png"),
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close()


def plot_scores_2d(scores_calib, scores_test, n_rows, n_faults, fault_names, pca):
    """2D scatter plot of first two PCs for calibration and test faults."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calibration
    ax.scatter(
        scores_calib[:, 0],
        scores_calib[:, 1],
        c="lightblue",
        s=3,
        alpha=0.3,
        label="Calibration",
    )

    # Test faults - distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, n_faults))
    for f in range(n_faults):
        start = f * n_rows
        end = (f + 1) * n_rows
        ax.scatter(
            scores_test[start:end, 0],
            scores_test[start:end, 1],
            c=[colors[f]],
            s=8,
            alpha=0.6,
            label=fault_names[f].strip(),
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize=12)
    ax.set_title("PCA Score Plot (PC1 vs PC2)", fontsize=13)
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, "pca_scores_2d.png"), dpi=DPI, bbox_inches="tight"
    )
    plt.close()


# ── Main execution ─────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("Semiconductor Anomaly Detection - PCA-based Analysis")
    print("LAM 9600 Metal Etcher Dataset")
    print("=" * 70)

    create_dirs()

    # 1. Load data
    print("\n[1/9] Loading data...")
    calibration, test, calib_names, test_names, fault_names, variables = load_data()
    n_calib = len(calibration)
    n_test_wafers = len(test)
    n_vars = len(variables)
    print(
        f"  Calibration: {n_calib} wafers, shapes: {calibration[0].shape[0]}-{calibration[-1].shape[0]} rows"
    )
    print(
        f"  Test: {n_test_wafers} wafers, shapes: {min(w.shape[0] for w in test)}-{max(w.shape[0] for w in test)} rows"
    )
    print(f"  Variables: {n_vars}")
    print(f"  Fault types: {fault_names}")

    # 2. Preprocess
    print("\n[2/9] Preprocessing (stacking time steps, normalizing)...")
    X_calib, X_test, scaler, n_rows, _, _ = preprocess(calibration, test)
    print(
        f"  Calibration samples: {X_calib.shape[0]} ({n_calib} wafers x {n_rows} rows)"
    )
    print(f"  Test samples: {X_test.shape[0]} ({n_test_wafers} wafers x {n_rows} rows)")
    print(f"  Features: {X_calib.shape[1]}")

    # 3. PCA
    print("\n[3/9] Performing PCA...")
    (
        pca,
        scores_calib,
        eigenvalues,
        explained_ratio,
        cumulative_ratio,
        kaiser_n,
        T2_calib,
        SPE_calib,
        loadings,
    ) = perform_pca(X_calib, variables)

    # 4. Scree plot
    print("\n[4/9] Generating scree plot and sensitivity analysis...")
    plot_scree(eigenvalues, explained_ratio, cumulative_ratio, kaiser_n)
    plot_sensitivity_analysis(X_calib, variables)

    # 5. Compute control limits
    print("\n[5/9] Computing control limits...")
    n_calib_samples = X_calib.shape[0]
    T2_limit = compute_T2_limit(kaiser_n, n_calib_samples, ALPHA)
    SPE_limit, g_spe, h_spe = compute_SPE_limit(SPE_calib, ALPHA)
    print(f"  Components retained: {kaiser_n}")
    print(f"  Variance explained: {cumulative_ratio[kaiser_n - 1] * 100:.1f}%")
    print(f"  T² limit (alpha={ALPHA}): {T2_limit:.2f}")
    print(
        f"  SPE limit (alpha={ALPHA}): {SPE_limit:.2f} (g={g_spe:.4f}, h={h_spe:.2f})"
    )

    # 6. Compute T² and SPE for test data
    print("\n[6/9] Computing T² and SPE for test wafers...")
    T2_test, SPE_test, scores_test = compute_T2_SPE_test(
        X_test, pca, T2_limit, SPE_limit
    )

    T2_rates, SPE_rates = compute_detection_rates(
        T2_test, SPE_test, T2_limit, SPE_limit, n_rows, n_test_wafers
    )

    # 7. Contributions
    print("\n[7/9] Computing variable contributions...")
    T2_contrib, SPE_contrib = compute_contributions(
        X_test, pca, n_rows, n_test_wafers, n_test_wafers, variables
    )

    # 8. Benchmarks
    print("\n[8/9] Running benchmarks (Isolation Forest, One-Class SVM)...")
    iso_pred, svm_pred, iso_rates, svm_rates = run_benchmarks(
        X_calib, X_test, n_rows, n_test_wafers, fault_names
    )

    # 9. Generate plots
    print("\n[9/9] Generating plots...")
    plot_T2_SPE_charts(
        T2_calib,
        SPE_calib,
        T2_test,
        SPE_test,
        T2_limit,
        SPE_limit,
        n_rows,
        n_test_wafers,
        fault_names,
    )
    plot_individual_fault_charts(
        T2_test, SPE_test, T2_limit, SPE_limit, n_rows, fault_names
    )
    plot_contribution_bars(T2_contrib, SPE_contrib, fault_names, variables)
    plot_detection_comparison(fault_names, T2_rates, SPE_rates, iso_rates, svm_rates)
    plot_scores_2d(scores_calib, scores_test, n_rows, n_test_wafers, fault_names, pca)

    # ── Save results ───────────────────────────────────────────────────────
    print("\nSaving results...")

    # Summary table
    summary_rows = []
    for f in range(n_test_wafers):
        summary_rows.append(
            {
                "Fault Type": fault_names[f].strip(),
                "T² Detection Rate (%)": round(T2_rates[f] * 100, 1),
                "SPE Detection Rate (%)": round(SPE_rates[f] * 100, 1),
                "IF Detection Rate (%)": round(iso_rates[f] * 100, 1),
                "OCSVM Detection Rate (%)": round(svm_rates[f] * 100, 1),
            }
        )
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "detection_summary.csv"), index=False)

    # PCA model info
    pca_info = {
        "n_components_kaiser": int(kaiser_n),
        "n_components_90pct": int(np.searchsorted(cumulative_ratio, 0.90) + 1),
        "eigenvalues": eigenvalues[: kaiser_n + 5].tolist(),
        "explained_variance_ratio": explained_ratio[: kaiser_n + 5].tolist(),
        "cumulative_variance": cumulative_ratio[: kaiser_n + 5].tolist(),
        "T2_limit": float(T2_limit),
        "SPE_limit": float(SPE_limit),
        "SPE_g": float(g_spe),
        "SPE_h": float(h_spe),
        "alpha": ALPHA,
        "n_calibration_samples": int(n_calib_samples),
        "n_test_samples": int(X_test.shape[0]),
        "n_time_rows_per_wafer": int(n_rows),
        "variables": variables,
        "calibration_T2_mean": float(np.mean(T2_calib)),
        "calibration_T2_std": float(np.std(T2_calib)),
        "calibration_SPE_mean": float(np.mean(SPE_calib)),
        "calibration_SPE_std": float(np.std(SPE_calib)),
    }
    with open(os.path.join(RESULTS_DIR, "pca_model.json"), "w") as f:
        json.dump(pca_info, f, indent=2)

    # Contribution data
    contrib_data = {}
    for f in range(n_test_wafers):
        fname = fault_names[f].strip()
        contrib_data[fname] = {
            "T2_contributions": {
                variables[v]: round(float(T2_contrib[f, v]), 6) for v in range(n_vars)
            },
            "SPE_contributions": {
                variables[v]: round(float(SPE_contrib[f, v]), 6) for v in range(n_vars)
            },
            "top_T2_vars": sorted(
                variables, key=lambda v: T2_contrib[f, variables.index(v)], reverse=True
            )[:5],
            "top_SPE_vars": sorted(
                variables,
                key=lambda v: SPE_contrib[f, variables.index(v)],
                reverse=True,
            )[:5],
        }
    with open(os.path.join(RESULTS_DIR, "contributions.json"), "w") as f:
        json.dump(contrib_data, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print(
        f"\nPCA: {kaiser_n} components, {cumulative_ratio[kaiser_n - 1] * 100:.1f}% variance"
    )
    print(f"T² limit: {T2_limit:.2f}, SPE limit: {SPE_limit:.2f}")
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Plots saved to: {PLOTS_DIR}/")
    print("=" * 70)

    return pca_info, df_summary, contrib_data


if __name__ == "__main__":
    pca_info, df_summary, contrib_data = main()
