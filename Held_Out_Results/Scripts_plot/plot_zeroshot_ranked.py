import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs('figures_tables', exist_ok=True)

# ── Data ────────────────────────────────────────────────────────────────────────

BUSI_STD = {
    'OpenAI CLIP':          ('OpenAI CLIP',      'baseline_a'),
    'BiomedCLIP':           ('BiomedCLIP',        'baseline_b'),
    'CLIP +Txt Enc (std)':  ('CLIP (+Text)',      'clip'),
    'CLIP MLP Only (std)':  ('CLIP (MLP only)',   'clip'),
    'CLIP +Img Enc (std)':  ('CLIP (+Image)',     'clip'),
    'CLIP Full (std)':      ('CLIP (Full FT)',    'clip'),
    'BERT Full (std)':      ('BERT (Full FT)',    'bert'),
    'BERT +Txt Enc (std)':  ('BERT (+Text)',      'bert'),
    'BERT MLP Only (std)':  ('BERT (MLP only)',   'bert'),
    'BERT +Img Enc (std)':  ('BERT (+Image)',     'bert'),
}

LIVER_STD = {
    'OpenAI CLIP':                                  ('OpenAI CLIP',      'baseline_a'),
    'BiomedCLIP':                                   ('BiomedCLIP',        'baseline_b'),
    'CLIP standard | MLP + image encoder':          ('CLIP (+Image)',     'clip'),
    'CLIP standard | MLP + text encoder':           ('CLIP (+Text)',      'clip'),
    'CLIP standard | MLP heads only':               ('CLIP (MLP only)',   'clip'),
    'CLIP standard | MLP + image + text encoder':   ('CLIP (Full FT)',    'clip'),
    'BERT standard | MLP heads only':               ('BERT (MLP only)',   'bert'),
    'BERT standard | MLP + image + text encoder':   ('BERT (Full FT)',    'bert'),
    'BERT standard | MLP + text encoder':           ('BERT (+Text)',      'bert'),
    'BERT standard | MLP + image encoder':          ('BERT (+Image)',     'bert'),
}

# ── Colors ──────────────────────────────────────────────────────────────────────
# Two tones: neutral gray (baselines) and vivid (trained)
COLORS = {
    'baseline_a': '#525252',   # dark grey  – OpenAI CLIP
    'baseline_b': '#2d6a4f',   # dark green – BiomedCLIP
    'clip':       '#2171b5',   # vivid blue  – all CLIP-Std trained
    'bert':       '#cb4b16',   # vivid burnt-orange – all BERT-Std trained
}

LEGEND_HANDLES = [
    mpatches.Patch(color=COLORS['baseline_a'], label='OpenAI CLIP (baseline)'),
    mpatches.Patch(color=COLORS['baseline_b'], label='BiomedCLIP (baseline)'),
    mpatches.Patch(color=COLORS['clip'],        label='CLIP encoder (trained)'),
    mpatches.Patch(color=COLORS['bert'],        label='BERT encoder (trained)'),
]


def prepare(raw_df, name_map):
    df = raw_df[raw_df['model'].isin(name_map)].copy()
    df['label'] = df['model'].map(lambda m: name_map[m][0])
    df['group'] = df['model'].map(lambda m: name_map[m][1])
    df['color'] = df['group'].map(COLORS)
    return df.sort_values('accuracy', ascending=True).reset_index(drop=True)


b_raw = pd.read_csv('BUSI_HO/classification_results.csv')
l_raw = pd.read_csv('Liver_HO/classification_results_final.csv')

b_df = prepare(b_raw, BUSI_STD)
l_df = prepare(l_raw, LIVER_STD)

# ── Plot ────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

for ax, df, panel in zip(axes, [b_df, l_df], ['(a) Breast', '(b) Liver']):
    bars = ax.barh(df['label'], df['accuracy'], color=df['color'],
                   edgecolor='white', linewidth=0.6, height=0.65)

    # Value labels at end of each bar
    for bar, val in zip(bars, df['accuracy']):
        ax.text(val + 0.007, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha='left', fontsize=8)

    ax.set_xlabel('Zero-Shot Classification Accuracy', fontsize=9)
    ax.set_ylabel('Model', fontsize=9)
    ax.set_xlim(0, max(df['accuracy']) + 0.12)
    ax.tick_params(axis='both', labelsize=8.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.12, 1.04, panel, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')

# Single shared legend below both panels
fig.legend(handles=LEGEND_HANDLES, loc='lower center', ncol=4,
           fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.06))

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('figures_tables/zeroshot_ranked.pdf', bbox_inches='tight', dpi=200)
plt.savefig('figures_tables/zeroshot_ranked.png', bbox_inches='tight', dpi=200)
plt.close()
print('Saved figures_tables/zeroshot_ranked.pdf and .png')
