"""
Sri Lanka Heat IBF — Demonstration Script
==========================================
Runs the full Bayesian warning system pipeline and produces
publication-quality charts saved to ./demo_output/.

Data modes (selected automatically):
  1. CSV file    — pass --csv path/to/data.csv
                   Expected columns: date, heat_index, [forecast_p1, forecast_p2, forecast_p3, forecast_p4]
                   (forecast columns optional; uniform ensemble assumed if absent)
  2. Seasonal    — realistic synthetic full-year with Mar–May heat peak
  3. Built-in    — plain synthetic from generate_demo_data()

Usage:
    python heat_ews_demo.py                   # built-in synthetic
    python heat_ews_demo.py --seasonal        # full-year seasonal cycle
    python heat_ews_demo.py --csv my_data.csv # real data

Requires: numpy, matplotlib, (pandas — only for CSV mode)
"""

import argparse, sys, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

# ── Import the heat EWS module from same directory ─────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat_ews_srilanka import (
    HeatData, HeatWarningSystem, ClimatologicalModel, CalibratedModel, EnsembleModel,
    build_loss_matrix, bayes_warning, brier_scores, hit_rates,
    generate_demo_data, categorise_hi, STATE_LABELS, WARNING_LABELS,
    HI_THRESHOLDS, SECTOR_LOSS_PARAMS, SECTOR_ACTIONS
)

# ── Style ───────────────────────────────────────────────────────
COLORS = {
    1: ('#2E7D32', '#E8F5E9'),   # green
    2: ('#F9A825', '#FFFDE7'),   # yellow
    3: ('#E65100', '#FFF3E0'),   # amber
    4: ('#B71C1C', '#FFEBEE'),   # red
}
WARN_SHORT = {1: 'No Warning', 2: 'Watch', 3: 'Alert', 4: 'Warning'}
SECTOR_COLORS = {
    'gf'       : '#455A64',   # blue-grey  — Tier 1 General Forecaster
    'balanced' : '#1565C0',   # deep blue  — Tier 2 Balanced
    'health'   : '#C62828',   # deep red
    'labour'   : '#E65100',   # deep orange
    'dm'       : '#6A1B9A',   # purple     — Disaster Mgmt
    'garment'  : '#00838F',   # dark cyan  — Garment & Apparel
    'tourism'  : '#00695C',   # teal
    'energy'   : '#F57F17',   # amber
    'agri'     : '#2E7D32',   # green      — excluded from Balanced
    'education': '#0D47A1',   # indigo     — excluded from Balanced
}
SECTOR_LABELS = {
    'gf'       : 'General Forecaster (T1)',
    'balanced' : 'Balanced Sectoral (T2)',
    'health'   : 'Human Health',
    'labour'   : 'Outdoor Labour',
    'dm'       : 'Disaster Mgmt',
    'garment'  : 'Garment & Apparel',
    'tourism'  : 'Tourism',
    'energy'   : 'Energy',
    'agri'     : 'Agriculture (excl.)',
    'education': 'Education (excl.)',
}
# Sectors included in Tier 2 Balanced calibration
BALANCED_SECTORS = ['health', 'labour', 'dm', 'garment', 'tourism', 'energy']
# All Tier 3 individual sector keys (for charts)
TIER3_SECTORS = ['health', 'labour', 'dm', 'garment', 'tourism', 'energy', 'agri', 'education']

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
})

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_output')
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# DATA GENERATORS
# ═══════════════════════════════════════════════════════════════

def generate_seasonal_data(seed=42):
    """Full-year synthetic dataset with realistic Colombo coastal seasonal cycle.

    Climatological pattern for Colombo (coastal, high-humidity urban):
      Jan     : warm and dry, moderate HI (Cat 1-2)
      Feb-Apr : inter-monsoonal heat peak; high humidity amplifies HI (Cat 2-4)
      May-Sep : SW Monsoon; rain and cloud reduce temperatures but humidity stays
                high — HI remains elevated for outdoor workers (Cat 2-3)
      Oct-Nov : NE Monsoon onset; short rains, moderate HI (Cat 1-2)
      Dec     : cooler dry period (Cat 1)

    Unlike the dry-zone interior, Colombo's Heat Index is driven as much by
    high relative humidity (70-90% year-round) as by Tmax. The baseline is
    lower than Colombo (~34°C vs 38°C) but the humidity keeps HI
    categories elevated across more months of the year.

    Returns train_data (days 1–270), test_data (days 271–365), dates list.
    """
    rng = np.random.default_rng(seed)
    n = 365
    doy = np.arange(1, n + 1)

    # Seasonal heat index signal tuned to Colombo coastal urban profile
    # Peak Feb-Apr (doy ~60-110), SW Monsoon partial cooling May-Sep,
    # brief NE Monsoon dip Oct-Nov, mild Dec-Jan
    hi_signal = (
        34.0                                              # lower baseline than dry zone
        + 7.0 * np.sin(2 * np.pi * (doy - 45) / 365)    # annual cycle peaking ~Apr
        + 5.0 * np.exp(-0.5 * ((doy - 80) / 25)**2)     # Feb-Apr sharp inter-monsoonal peak
        - 2.0 * np.exp(-0.5 * ((doy - 210) / 50)**2)    # partial SW Monsoon cooling
        + 1.5 * np.sin(2 * np.pi * doy / 182)            # semi-annual humidity modulation
        + rng.normal(0, 1.8, n)                           # daily noise (lower variance, coastal)
    )

    obs_cats = np.array([categorise_hi(float(h)) for h in hi_signal])

    # Build forecast probability vectors: ensemble with moderate skill
    probs = np.zeros((n, 4))
    for t in range(n):
        x = obs_cats[t]
        base = np.zeros(4)
        base[x - 1] = 0.55
        base += 0.12
        probs[t] = rng.dirichlet(base * 6 + 1)

    # Build ISO date strings
    import datetime
    base_date = datetime.date(2026, 1, 1)
    dates = [(base_date + datetime.timedelta(days=i)).isoformat() for i in range(n)]

    train = HeatData(obs_cats[:270], probs[:270], dates[:270])
    test  = HeatData(obs_cats[270:], probs[270:], dates[270:])
    return train, test, hi_signal, dates


def load_csv_data(csv_path):
    """Load real data from a CSV file.

    Expected columns:
      date           : ISO format (YYYY-MM-DD)
      heat_index     : observed Heat Index (°C) — used to derive category
      forecast_p1    : forecast probability for Cat 1 (optional)
      forecast_p2    : forecast probability for Cat 2 (optional)
      forecast_p3    : forecast probability for Cat 3 (optional)
      forecast_p4    : forecast probability for Cat 4 (optional)

    If forecast columns are absent, a uniform ensemble centred on the
    observed category is used (moderate-skill assumption).
    """
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required to load CSV data. Install with: pip install pandas")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    required = ['date', 'heat_index']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    obs_cats = np.array([categorise_hi(float(h)) for h in df['heat_index']])
    n = len(obs_cats)
    dates = df['date'].astype(str).tolist()

    has_forecast = all(f'forecast_p{j}' in df.columns for j in range(1, 5))
    if has_forecast:
        probs = df[['forecast_p1','forecast_p2','forecast_p3','forecast_p4']].values.astype(float)
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = np.where(row_sums > 0, probs / row_sums, 0.25)
        print(f"  Loaded {n} days from CSV with forecast probability columns.")
    else:
        rng = np.random.default_rng(0)
        probs = np.zeros((n, 4))
        for t in range(n):
            x = obs_cats[t]
            base = np.zeros(4); base[x-1] = 0.50; base += 0.12
            probs[t] = rng.dirichlet(base * 5 + 1)
        print(f"  Loaded {n} days from CSV (no forecast columns — synthetic ensemble assumed).")

    split = int(n * 0.75)
    train = HeatData(obs_cats[:split], probs[:split], dates[:split])
    test  = HeatData(obs_cats[split:], probs[split:], dates[split:])
    return train, test


# ═══════════════════════════════════════════════════════════════
# CHART 1 — Seasonal Warning Calendar
# ═══════════════════════════════════════════════════════════════

def chart_seasonal_calendar(test_data, issued_warnings, hi_signal=None, dates=None, suffix=''):
    """Timeline of warning levels issued over the evaluation period."""
    n = test_data.n
    x = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1.2, 1]})
    fig.suptitle('Sri Lanka Heat IBF — Colombo Pilot\nWarning Calendar & Seasonal Pattern',
                 fontsize=14, fontweight='bold', y=0.98)

    # ── Panel 1: Warning level bars ──────────────────────────
    ax = axes[0]
    for t in range(n):
        w = issued_warnings[t]
        ax.bar(t, 1, color=COLORS[w][0], edgecolor='none', linewidth=0)
    ax.set_yticks([])
    ax.set_ylabel('Warning level', fontsize=11)
    ax.set_ylim(0, 1)

    # Legend patches
    patches = [mpatches.Patch(color=COLORS[w][0], label=WARN_SHORT[w]) for w in [1,2,3,4]]
    ax.legend(handles=patches, loc='upper right', ncol=4, framealpha=0.9)
    ax.set_title('Issued Warning Level (Calibrated Bayesian System)', loc='left')

    # Annotate month boundaries if dates available
    if dates:
        import datetime
        month_starts = {}
        for t, d in enumerate(dates):
            m = datetime.date.fromisoformat(d).month
            if m not in month_starts:
                month_starts[m] = t
        month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                       7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        for m, pos in month_starts.items():
            ax.axvline(pos, color='white', linewidth=0.8, alpha=0.5)
            ax.text(pos + 1, 0.92, month_names.get(m,''), fontsize=8, color='white', alpha=0.9)

    # ── Panel 2: Forecast probability of high heat ───────────
    ax2 = axes[1]
    p_high = test_data.forecast_probs[:, 2] + test_data.forecast_probs[:, 3]
    ax2.fill_between(x, 0, p_high, color='#E65100', alpha=0.4, label='P(Cat 3+4) forecast')
    ax2.plot(x, p_high, color='#E65100', linewidth=0.8, alpha=0.7)
    ax2.axhline(0.25, color='#F9A825', linewidth=1.5, linestyle='--', label='Watch trigger (~25%)')
    ax2.axhline(0.50, color='#B71C1C', linewidth=1.5, linestyle='--', label='Alert trigger (~50%)')
    ax2.set_ylabel('P(heat extreme)', fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_title('Ensemble Forecast: Probability of Extreme/Danger Heat', loc='left')

    # ── Panel 3: Observed category ───────────────────────────
    ax3 = axes[2]
    for t in range(n):
        c = test_data.observed_categories[t]
        ax3.bar(t, c, color=COLORS[c][0], alpha=0.85, edgecolor='none')
    ax3.set_yticks([1,2,3,4])
    ax3.set_yticklabels(['Normal','Caution','Extreme','Danger'], fontsize=9)
    ax3.set_ylabel('Observed category', fontsize=11)
    ax3.set_xlabel('Day in evaluation period', fontsize=11)
    ax3.set_title('Observed Heat Severity Category', loc='left')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'1_seasonal_calendar{suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# CHART 2 — Model Probability Comparison
# ═══════════════════════════════════════════════════════════════

def chart_probability_comparison(train_data, test_data, suffix=''):
    """Compare p(x3+x4|y) across CLIM, CAL, ENS for each test day."""
    clim = ClimatologicalModel().fit(train_data)
    cal  = CalibratedModel().fit(train_data)
    ens  = EnsembleModel()

    n = min(test_data.n, 120)   # show first 120 test days for clarity

    clim_ph = np.array([clim.predict(test_data.forecast_probs[t])[2:].sum() for t in range(n)])
    cal_ph  = np.array([cal.predict(test_data.forecast_probs[t])[2:].sum()  for t in range(n)])
    ens_ph  = np.array([ens.predict(test_data.forecast_probs[t])[2:].sum()  for t in range(n)])
    obs_high = (test_data.observed_categories[:n] >= 3).astype(float)

    x = np.arange(n)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle('Probability Model Comparison\nP(Extreme or Danger Heat) — three models vs observations',
                 fontsize=14, fontweight='bold')

    labels_models = [('Climatological', clim_ph, '#90A4AE'),
                     ('Calibrated (CAL)', cal_ph, '#1565C0'),
                     ('Raw Ensemble (ENS)', ens_ph, '#E65100')]

    for ax, (name, ph, color) in zip(axes, labels_models):
        ax.fill_between(x, 0, obs_high, color='#B71C1C', alpha=0.15, label='Observed high heat')
        ax.step(x, ph, color=color, linewidth=1.5, label=f'{name} forecast', where='mid')
        ax.axhline(0.5, color='grey', linewidth=0.8, linestyle=':')
        ax.set_ylim(-0.05, 1.15)
        ax.set_ylabel('Probability', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(f'{name}', loc='left', fontsize=11)

    axes[-1].set_xlabel('Day in evaluation period', fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'2_probability_models{suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# CHART 3 — Brier Score and POD/FAR Comparison
# ═══════════════════════════════════════════════════════════════

def chart_verification(train_data, test_data, suffix=''):
    """Brier scores per category and POD/FAR for all three model types."""
    results = {}
    for model_type in ['climatological', 'calibrated', 'ensemble']:
        sys = HeatWarningSystem(sector='balanced', model_type=model_type)
        if model_type != 'ensemble':
            sys.fit(train_data)
        r = sys.evaluate(test_data)
        bs = brier_scores(test_data, sys.model)
        hr = hit_rates(test_data, sys.loss_matrix, sys.model)
        results[model_type] = {'bs': bs, 'hr': hr, 'loss': r['total_loss']}

    cats = ['Cat 1\n(Normal)', 'Cat 2\n(Caution)', 'Cat 3\n(Extreme)', 'Cat 4\n(Danger)']
    models = list(results.keys())
    model_labels = ['Climatological', 'Calibrated (CAL)', 'Raw Ensemble']
    model_colors = ['#90A4AE', '#1565C0', '#E65100']
    x = np.arange(4)
    width = 0.25

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle('Verification Statistics — Model Comparison\nColombo Pilot Evaluation Period',
                 fontsize=14, fontweight='bold')

    # ── Brier scores ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for i, (mt, label, color) in enumerate(zip(models, model_labels, model_colors)):
        ax1.bar(x + i*width, results[mt]['bs'], width, label=label,
                color=color, alpha=0.85, edgecolor='white')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(cats)
    ax1.set_ylabel('Brier Score (lower = better)')
    ax1.set_title('Brier Score by Severity Category', loc='left')
    ax1.legend(loc='upper left')
    ax1.axhline(0.25, color='grey', linewidth=0.8, linestyle=':', label='Random baseline')

    # ── POD / FAR ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    pod_vals = [results[mt]['hr']['POD'] for mt in models]
    far_vals = [results[mt]['hr']['FAR'] for mt in models]
    bars = ax2.bar(model_labels, pod_vals, color=model_colors, alpha=0.85, edgecolor='white')
    ax2.set_ylabel('Probability of Detection (POD)')
    ax2.set_title('POD — High Heat Events (Cat 3+4)', loc='left')
    ax2.set_ylim(0, 1.1)
    for bar, val in zip(bars, pod_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax3 = fig.add_subplot(gs[1, 1])
    bars2 = ax3.bar(model_labels, far_vals, color=model_colors, alpha=0.85, edgecolor='white')
    ax3.set_ylabel('False Alarm Rate (FAR)')
    ax3.set_title('FAR — High Heat Events (lower = better)', loc='left')
    ax3.set_ylim(0, 1.1)
    ax3.axhline(0.30, color='#F9A825', linewidth=1.5, linestyle='--', label='Target FAR < 0.30')
    ax3.legend(fontsize=9)
    for bar, val in zip(bars2, far_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'3_verification{suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# CHART 4 — Cumulative Loss Curves
# ═══════════════════════════════════════════════════════════════

def chart_loss_curves(train_data, test_data, suffix=''):
    """Cumulative loss over time for each model × sector."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Cumulative Loss Curves — Bayesian System vs Baselines\n'
                 '(Lower = better; loss units are relative)',
                 fontsize=14, fontweight='bold')

    # Panel 1: model comparison (balanced sector)
    ax = axes[0]
    for model_type, label, color, ls in [
        ('climatological', 'Climatological (baseline)', '#90A4AE', ':'),
        ('ensemble',       'Raw Ensemble',              '#F9A825', '--'),
        ('calibrated',     'Calibrated Bayesian (CAL)', '#1565C0', '-'),
    ]:
        sys = HeatWarningSystem(sector='balanced', model_type=model_type)
        if model_type != 'ensemble':
            sys.fit(train_data)
        r = sys.evaluate(test_data)
        cum = np.cumsum(r['actual_losses'])
        ax.plot(cum, label=label, color=color, linestyle=ls, linewidth=2)

    ax.set_xlabel('Day in evaluation period')
    ax.set_ylabel('Cumulative loss')
    ax.set_title('Model comparison\n(balanced sector loss)', loc='left')
    ax.legend(loc='upper left')

    # Panel 2: three-tier sector comparison (calibrated model)
    ax2 = axes[1]
    # Tier 1 and Tier 2 as bold reference lines
    for skey, ls, lw in [('gf', '--', 2.5), ('balanced', '-', 2.5)]:
        s = HeatWarningSystem(sector=skey, model_type='calibrated')
        s.fit(train_data)
        r2 = s.evaluate(test_data)
        ax2.plot(np.cumsum(r2['actual_losses']), label=SECTOR_LABELS[skey],
                 color=SECTOR_COLORS[skey], linestyle=ls, linewidth=lw)
    # Tier 3 individual sectors
    for sector in TIER3_SECTORS:
        s = HeatWarningSystem(sector=sector, model_type='calibrated')
        s.fit(train_data)
        r2 = s.evaluate(test_data)
        lw = 1.5 if sector in ('agri', 'education') else 2.0
        ax2.plot(np.cumsum(r2['actual_losses']),
                 label=SECTOR_LABELS[sector],
                 color=SECTOR_COLORS[sector],
                 linewidth=lw,
                 linestyle=':' if sector in ('agri', 'education') else '-')

    ax2.set_xlabel('Day in evaluation period')
    ax2.set_ylabel('Cumulative loss')
    ax2.set_title('Three-tier sector loss profiles\n(calibrated model  |  dotted = excluded from Balanced)',
                  loc='left')
    ax2.legend(loc='upper left', fontsize=8, ncol=2)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'4_loss_curves{suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# CHART 5 — Warning Decision Heatmap
# ═══════════════════════════════════════════════════════════════

def chart_decision_heatmap(train_data, suffix=''):
    """Show optimal warning for a grid of forecast probability scenarios."""
    sys = HeatWarningSystem(sector='balanced', model_type='calibrated')
    sys.fit(train_data)

    # Axes: P(Cat 4 = Danger) vs P(Cat 3 = Extreme)
    p3_vals = np.linspace(0, 0.6, 25)
    p4_vals = np.linspace(0, 0.6, 25)
    grid = np.zeros((len(p4_vals), len(p3_vals)), dtype=int)

    for i, p4 in enumerate(p4_vals):
        for j, p3 in enumerate(p3_vals):
            p2 = max(0, min(0.5, 1 - p3 - p4) * 0.5)
            p1 = max(0, 1 - p2 - p3 - p4)
            forecast = np.array([p1, p2, p3, p4])
            forecast /= forecast.sum()
            grid[i, j] = sys.issue_warning(forecast)['warning']

    cmap = matplotlib.colors.ListedColormap(
        [COLORS[w][0] for w in [1,2,3,4]])
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Optimal Warning Decision Space\nBayesian Calibrated Model — Balanced Sector Loss',
                 fontsize=14, fontweight='bold')

    # Main heatmap
    ax = axes[0]
    im = ax.pcolormesh(p3_vals, p4_vals, grid, cmap=cmap, norm=norm)
    ax.set_xlabel('P(Cat 3 — Extreme Caution)', fontsize=11)
    ax.set_ylabel('P(Cat 4 — Danger)', fontsize=11)
    ax.set_title('Optimal warning level by forecast probability', loc='left')
    patches = [mpatches.Patch(color=COLORS[w][0], label=f'Level {w}: {WARN_SHORT[w]}') for w in [1,2,3,4]]
    ax.legend(handles=patches, loc='upper left', fontsize=9)

    # Contour overlay showing transition boundaries
    ax.contour(p3_vals, p4_vals, grid, levels=[1.5, 2.5, 3.5],
               colors='white', linewidths=2, linestyles='--')

    # Panel 2: Three-tier sector comparison along P(extreme) axis
    ax2 = axes[1]
    p3_scan = np.linspace(0, 0.5, 50)
    # Tier 1 + Tier 2 as bold reference lines
    for skey, ls, lw in [('gf', '--', 2.8), ('balanced', '-', 2.8)]:
        s = HeatWarningSystem(sector=skey, model_type='calibrated')
        s.fit(train_data)
        warn_levels = []
        for p3 in p3_scan:
            p4 = p3 * 0.5
            p1 = max(0, 1 - p3 - p4) * 0.6
            p2 = max(0, 1 - p1 - p3 - p4)
            fc = np.array([p1, p2, p3, p4]); fc /= fc.sum()
            warn_levels.append(s.issue_warning(fc)['warning'])
        ax2.step(p3_scan, warn_levels, where='mid',
                 color=SECTOR_COLORS[skey], linewidth=lw, linestyle=ls,
                 label=SECTOR_LABELS[skey], zorder=3)
    # Tier 3 sectors
    for sector in TIER3_SECTORS:
        s = HeatWarningSystem(sector=sector, model_type='calibrated')
        s.fit(train_data)
        warn_levels = []
        for p3 in p3_scan:
            p4 = p3 * 0.5
            p1 = max(0, 1 - p3 - p4) * 0.6
            p2 = max(0, 1 - p1 - p3 - p4)
            fc = np.array([p1, p2, p3, p4]); fc /= fc.sum()
            warn_levels.append(s.issue_warning(fc)['warning'])
        ls2 = ':' if sector in ('agri', 'education') else '-'
        ax2.step(p3_scan, warn_levels, where='mid',
                 color=SECTOR_COLORS[sector], linewidth=1.8, linestyle=ls2,
                 label=SECTOR_LABELS[sector], alpha=0.8)

    ax2.set_yticks([1,2,3,4])
    ax2.set_yticklabels([WARN_SHORT[w] for w in [1,2,3,4]])
    ax2.set_xlabel('P(Cat 3 — Extreme heat) in forecast', fontsize=11)
    ax2.set_ylabel('Issued warning level', fontsize=11)
    ax2.set_title('Three-tier sector comparison along P(extreme) axis\n'
                  '(P(danger) = 0.5×P(extreme)  |  dotted = excluded from Balanced)', loc='left')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.set_ylim(0.5, 4.5)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'5_decision_heatmap{suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# CHART 6 — Single Forecast Advisory Panel
# ═══════════════════════════════════════════════════════════════

def chart_advisory_panel(train_data, forecast_p, scenario_label='Scenario A', suffix=''):
    """Three-tier advisory panel for a specific forecast.

    Uses the ensemble model (raw forecast probabilities) so that sector
    differentiation is driven directly by the forecast vector rather than
    being absorbed by calibration on the synthetic training data.  When
    real DoM observation data are loaded, switch to model_type='calibrated'.

    Shows Tier 1 Hazard Warning (GF), Tier 2 Impact Advisory (Balanced), and Tier 3 Decision Advisory bar alongside
    calibrated probabilities and the Tier 2 sector action table.
    """
    # ── Compute all three tiers (ensemble model preserves forecast signal) ─
    sys_gf = HeatWarningSystem(sector='gf', model_type='ensemble')
    r_gf = sys_gf.issue_warning(forecast_p)
    w_gf = r_gf['warning']

    sys_bal = HeatWarningSystem(sector='balanced', model_type='ensemble')
    r_bal = sys_bal.issue_warning(forecast_p)
    w_bal = r_bal['warning']

    t3_warnings = {}
    for skey in TIER3_SECTORS:
        s = HeatWarningSystem(sector=skey, model_type='ensemble')
        t3_warnings[skey] = s.issue_warning(forecast_p)['warning']

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(COLORS[w_bal][1])
    gs = GridSpec(3, 3, figure=fig, hspace=0.6, wspace=0.42)
    fig.suptitle(f'Sri Lanka Heat IBF — Three-Tier Advisory\n{scenario_label}',
                 fontsize=14, fontweight='bold', y=0.99)

    # ── Row 0: Tier 1 box ─────────────────────────────────
    ax_t1 = fig.add_subplot(gs[0, 0])
    ax_t1.set_facecolor(COLORS[w_gf][0])
    ax_t1.text(0.5, 0.65, 'TIER 1 — GENERAL FORECASTER', transform=ax_t1.transAxes,
               fontsize=8.5, color='white', ha='center', alpha=0.85, va='center')
    ax_t1.text(0.5, 0.35, WARNING_LABELS[w_gf].upper(), transform=ax_t1.transAxes,
               fontsize=14, fontweight='bold', color='white', ha='center', va='center')
    ax_t1.text(0.5, 0.12, 'p* ≈ 0.50  |  pure meteorology',
               transform=ax_t1.transAxes, fontsize=8, color='white', ha='center', alpha=0.8)
    ax_t1.set_xticks([]); ax_t1.set_yticks([])
    for spine in ax_t1.spines.values(): spine.set_visible(False)

    # ── Row 0: Tier 2 box ─────────────────────────────────
    ax_t2 = fig.add_subplot(gs[0, 1])
    ax_t2.set_facecolor(COLORS[w_bal][0])
    ax_t2.text(0.5, 0.65, 'TIER 2 — BALANCED SECTORAL', transform=ax_t2.transAxes,
               fontsize=8.5, color='white', ha='center', alpha=0.85, va='center')
    ax_t2.text(0.5, 0.35, WARNING_LABELS[w_bal].upper(), transform=ax_t2.transAxes,
               fontsize=14, fontweight='bold', color='white', ha='center', va='center')
    ax_t2.text(0.5, 0.12, 'p* ≈ 0.40  |  5 heat-critical sectors',
               transform=ax_t2.transAxes, fontsize=8, color='white', ha='center', alpha=0.8)
    ax_t2.set_xticks([]); ax_t2.set_yticks([])
    for spine in ax_t2.spines.values(): spine.set_visible(False)

    # ── Row 0: Expected loss bar (Tier 2) ─────────────────
    ax_el = fig.add_subplot(gs[0, 2])
    ax_el.set_facecolor('white')
    el = list(r_bal['expected_losses'].values())
    bars = ax_el.barh([WARN_SHORT[i+1] for i in range(4)], el,
                      color=[COLORS[i+1][0] for i in range(4)], alpha=0.85)
    bars[w_bal-1].set_edgecolor('black'); bars[w_bal-1].set_linewidth(2)
    ax_el.set_xlabel('Expected loss')
    ax_el.set_title('Tier 2 Impact Advisory — expected loss', fontsize=10, fontweight='bold')
    ax_el.invert_yaxis()
    ax_el.set_facecolor('#FAFAFA')

    # ── Row 1: Calibrated probabilities ───────────────────
    ax_probs = fig.add_subplot(gs[1, 0])
    ax_probs.set_facecolor('white')
    pvals = list(r_bal['p_by_state'].values())
    ax_probs.bar(['Normal', 'Caution', 'Extreme', 'Danger'], pvals,
                 color=[COLORS[i+1][0] for i in range(4)], alpha=0.85, edgecolor='white')
    ax_probs.set_ylabel('Probability')
    ax_probs.set_title('Calibrated forecast\nprobabilities', fontsize=10, fontweight='bold')
    ax_probs.set_ylim(0, 1.05)
    ax_probs.tick_params(axis='x', labelsize=9)
    ax_probs.set_facecolor('#FAFAFA')

    # ── Row 1: Tier 3 sector comparison bar ───────────────
    ax_t3 = fig.add_subplot(gs[1, 1:])
    ax_t3.set_facecolor('white')
    ax_t3.set_title('Tier 3 — Decision Advisory by sector', fontsize=10, fontweight='bold')
    tick_labels = [SECTOR_LABELS[k].replace(' (excl.)', '*') for k in TIER3_SECTORS]
    bar_c = [COLORS[t3_warnings[k]][0] for k in TIER3_SECTORS]
    ax_t3.bar(tick_labels, [t3_warnings[k] for k in TIER3_SECTORS],
              color=bar_c, alpha=0.85)
    ax_t3.axhline(w_gf,  color=SECTOR_COLORS['gf'],      linewidth=2, linestyle='--',
                  label=f'Tier 1 GF ({WARN_SHORT[w_gf]})')
    ax_t3.axhline(w_bal, color=SECTOR_COLORS['balanced'], linewidth=2, linestyle='-',
                  label=f'Tier 2 Balanced ({WARN_SHORT[w_bal]})')
    ax_t3.set_yticks([1, 2, 3, 4])
    ax_t3.set_yticklabels(['Green', 'Yellow', 'Amber', 'Red'])
    ax_t3.set_ylabel('Optimal warning level')
    ax_t3.tick_params(axis='x', labelsize=8, rotation=20)
    ax_t3.legend(fontsize=9)
    ax_t3.text(6.5, 0.75, '* excl. from\n  Balanced', fontsize=7, color='#666',
               ha='right', va='bottom')

    # ── Row 2: Sector action table (Tier 2 Balanced) ──────
    ax_act = fig.add_subplot(gs[2, :])
    ax_act.axis('off')
    ax_act.set_facecolor('white')
    ax_act.set_title(f'Sector actions — Tier 2 Impact Advisory: {WARNING_LABELS[w_bal]}',
                     fontsize=10, fontweight='bold', loc='left', pad=6)
    sector_col_map = {
        'Health'   : '#C62828', 'Labour'   : '#E65100', 'DM'       : '#6A1B9A',
        'Tourism'  : '#00695C', 'Energy'   : '#F57F17', 'Agri'     : '#2E7D32',
        'Education': '#0D47A1',
    }
    col = 0; col_width = 0.33
    for sector, actions in SECTOR_ACTIONS[w_bal].items():
        color = sector_col_map.get(sector, '#333333')
        x_off = (col % 3) * col_width
        row_y = 0.97 - (col // 3) * 0.55
        ax_act.text(x_off, row_y, f'[{sector}]', transform=ax_act.transAxes,
                    fontsize=9, fontweight='bold', color=color, va='top')
        for k, action in enumerate(actions[:2]):
            ax_act.text(x_off + 0.01, row_y - 0.12*(k+1),
                        '• ' + action[:68] + ('…' if len(action) > 68 else ''),
                        transform=ax_act.transAxes, fontsize=7.5, color='#333333', va='top')
        col += 1

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'6_advisory_{suffix or "A"}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# CHART 7 — CSV Template / Data Summary
# ═══════════════════════════════════════════════════════════════

def write_csv_template():
    """Write a CSV template file for DoM to fill in real data."""
    import datetime
    base = datetime.date(2026, 3, 1)
    lines = ['date,heat_index,forecast_p1,forecast_p2,forecast_p3,forecast_p4']
    for i in range(7):
        d = (base + datetime.timedelta(days=i)).isoformat()
        lines.append(f'{d},,,,,    # example row — fill in observed HI and forecast probs')
    lines += [
        '',
        '# COLUMN DESCRIPTIONS:',
        '# date         : ISO format YYYY-MM-DD (required)',
        '# heat_index   : observed Heat Index in Celsius (required)',
        '#                  Normal <39  |  Caution 39-45  |  Extreme 46-51  |  Danger >=52',
        '# forecast_p1  : forecast probability of Cat 1 (Normal)  — optional',
        '# forecast_p2  : forecast probability of Cat 2 (Caution) — optional',
        '# forecast_p3  : forecast probability of Cat 3 (Extreme) — optional',
        '# forecast_p4  : forecast probability of Cat 4 (Danger)  — optional',
        '#   If omitted, a synthetic moderate-skill ensemble is assumed.',
        '#   Probabilities per row should sum to 1.0 (auto-normalised if not).',
        '',
        '# UTCI or Tmax users: convert to approximate Heat Index equivalents,',
        '# or adjust HI_THRESHOLDS in heat_ews_srilanka.py.',
    ]
    path = os.path.join(OUT_DIR, 'colombo_data_template.csv')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Sri Lanka Heat IBF — Demo')
    parser.add_argument('--seasonal', action='store_true',
                        help='Use full-year seasonal synthetic data')
    parser.add_argument('--csv', metavar='FILE',
                        help='Path to real observation CSV file')
    args = parser.parse_args()

    print('\n' + '='*60)
    print('  SRI LANKA HEAT IBF — DEMO')
    print('  Colombo District Pilot')
    print('='*60)

    hi_signal = None
    dates = None
    suffix = ''

    if args.csv:
        print(f'\nData mode: CSV — {args.csv}')
        train_data, test_data = load_csv_data(args.csv)
        suffix = '_csv'
    elif args.seasonal:
        print('\nData mode: Full-year seasonal synthetic cycle')
        train_data, test_data, hi_signal, dates = generate_seasonal_data()
        test_dates = dates[270:]
        suffix = '_seasonal'
    else:
        print('\nData mode: Built-in synthetic (generate_demo_data)')
        train_data, test_data = generate_demo_data()
        suffix = '_builtin'

    print(f'  Training days : {train_data.n}')
    print(f'  Evaluation days: {test_data.n}')

    # Fit calibrated system for charts
    sys_cal = HeatWarningSystem(sector='balanced', model_type='calibrated')
    sys_cal.fit(train_data)
    r = sys_cal.evaluate(test_data)
    issued = np.array(r['warning_issued'], dtype=int)

    print('\nGenerating charts...')
    chart_seasonal_calendar(test_data, issued,
                            hi_signal=hi_signal,
                            dates=test_dates if args.seasonal else None,
                            suffix=suffix)
    chart_probability_comparison(train_data, test_data, suffix=suffix)
    chart_verification(train_data, test_data, suffix=suffix)
    chart_loss_curves(train_data, test_data, suffix=suffix)
    chart_decision_heatmap(train_data, suffix=suffix)

    # Advisory panels for three representative scenarios.
    # Forecasts are chosen to produce clear sector differentiation:
    #   A: 3 distinct levels — Agri/Educ=Yellow, GF/Tourism/Energy=Amber,
    #      Balanced/Health/Labour/DM=Red.  Impact Advisory (Tier 2) is visibly more precautionary
    #      than Hazard Warning (Tier 1), justifying the three-tier architecture.
    #   B: Cost-sensitive lag — all heat-critical sectors escalate to Red while
    #      Agriculture/Education (excluded from Balanced) hold at Amber.
    #   C: Near-consensus critical heat — even Education reaches Red; only
    #      Agriculture stays one level lower due to its high mobilisation cost.
    scenarios = [
        ([0.45, 0.32, 0.15, 0.08], 'Scenario_A_Sector_Divergence'),
        ([0.20, 0.35, 0.28, 0.17], 'Scenario_B_Cost_Sensitive_Lag'),
        ([0.05, 0.10, 0.35, 0.50], 'Scenario_C_Critical_Heat'),
    ]
    for fp, label in scenarios:
        chart_advisory_panel(train_data, fp, scenario_label=label.replace('_', ' '), suffix=label)

    write_csv_template()

    print(f'\nAll outputs saved to: {OUT_DIR}/')
    print('\nSummary — Calibrated Bayesian System (Balanced sector):')
    print(f'  Total loss    : {r["total_loss"]:.0f}')
    print(f'  POD (high heat): {r["POD"]:.2f}')
    print(f'  FAR (high heat): {r["FAR"]:.2f}')
    print(f'  CSI (high heat): {r["CSI"]:.2f}')

    # Print warning distribution
    print('\nWarning frequency:')
    for w in [1,2,3,4]:
        count = np.sum(issued == w)
        pct = 100 * count / len(issued)
        bar = '█' * int(pct / 2)
        print(f'  {WARN_SHORT[w]:12s}: {count:3d} days ({pct:5.1f}%) {bar}')

    print('\nDone. Run with --seasonal or --csv to switch data modes.\n')


if __name__ == '__main__':
    main()
