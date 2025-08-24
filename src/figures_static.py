import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


def plot_national_fert_static(stacked_asfr, tfr_df):
    dist_names = (
        stacked_asfr
        .columns
        .to_series(name='col')
        .apply(lambda s: s.split('_', 1)[0])
    )
    prefixes = dist_names.unique().tolist()
    if len(prefixes) != 4:
        raise ValueError(f"Expected 4 distributions, got {prefixes}")

    # ─── 2. Compute μ, m, M for each age‐row i and each prefix d ────────────────
    means = pd.DataFrame(index=stacked_asfr.index, columns=prefixes, dtype=float)
    mins = means.copy()
    maxs = means.copy()

    for d in prefixes:
        cols = dist_names[dist_names == d].index
        sub = stacked_asfr[cols]  # (9 ages)×(|C_d| draws)
        means[d] = sub.mean(axis=1)
        mins[d] = sub.min(axis=1)
        maxs[d] = sub.max(axis=1)

    # ─── 3. Prepare bar positions ────────────────────────────────────────────────
    n_age = stacked_asfr.shape[0]  # 9
    n_dist = len(prefixes)  # 4
    y = np.arange(n_age)
    h = 0.8 / n_dist  # each distribution’s bar‐thickness

    age_labels = ['15–19', '20–24', '25–29', '30–34', '35–39',
                  '40–44', '45–49', '50–54', '55–59']

    # ─── 4. Plot both panels ────────────────────────────────────────────────────
    dist_labels = prefixes
    colors = ["#345995", "#B80C09", "#D4AF37", "#2E6F40"]  # one per prefix

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # — a) Horizontal‐bar + error‐bars on ax1
    for j, d in enumerate(dist_labels):
        μ = means[d].values
        lower = μ - mins[d].values
        upper = maxs[d].values - μ

        ax1.barh(
            y + j * h,
            μ,
            height=h,
            xerr=[lower, upper],
            capsize=3,
            color=colors[j],
            edgecolor='k',
            alpha=0.7
        )

    ax1.set_yticks(y + (n_dist - 1) / 2 * h)
    ax1.set_yticklabels(age_labels)
    ax1.set_xlabel('Age-Standardised Fertility Rate\n(National)')
    ax1.set_ylabel('Age Range')
    ax1.set_title('a.', loc='left', fontweight='bold', fontsize=15)
    ax1.grid(which='major', linestyle='--', alpha=0.2)

    # — b) KDE plots on ax2
    for j, d in enumerate(dist_labels):
        sns.kdeplot(
            data=tfr_df,
            x=d,
            ax=ax2,
            color=colors[j],
            linewidth=2,
            alpha=0.7
        )

    ax2.set_xlabel('Total Fertility Rate\n(National)')
    ax2.set_title('b.', loc='left', fontweight='bold', fontsize=15)
    ax2.grid(which='major', linestyle='--', alpha=0.2)

    sns.despine()
    plt.tight_layout()

    # ─── 5. Build custom legends ─────────────────────────────────────────────────
    # a) Bar legend via Patch
    bar_patches = [
        mpatches.Patch(facecolor=colors[i], edgecolor='k', alpha=0.7, label=dist_labels[i])
        for i in range(n_dist)
    ]
    ax1.legend(
        handles=bar_patches,
        title='Distribution',
        loc='upper right',
        frameon=True,
        edgecolor='k'
    )

    # b) KDE legend via Line2D
    kde_lines = [
        Line2D([0], [0], color=colors[i], lw=2, alpha=0.7, label=dist_labels[i])
        for i in range(n_dist)
    ]
    ax2.legend(
        handles=kde_lines,
        title='Distribution',
        loc='upper right',
        frameon=True,
        edgecolor='k'
    )
    fig.savefig('../figures/natioanl_fert_static.pdf')
