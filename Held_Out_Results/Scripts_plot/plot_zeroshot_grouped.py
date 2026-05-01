import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs('figures_tables', exist_ok=True)

# ── Color scheme (same tones as barplot 1, lighter shade = LLM variant) ─────────
COLORS = {
    'baseline_a':  '#525252',  # dark grey  – OpenAI CLIP
    'baseline_b':  '#2d6a4f',  # dark green – BiomedCLIP
    'clip_std':    '#2171b5',  # vivid blue  – CLIP std
    'clip_llm':    '#9ecae1',  # light blue  – CLIP LLM
    'bert_std':    '#cb4b16',  # vivid orange – BERT std
    'bert_llm':    '#fdae6b',  # light orange – BERT LLM
}

LEGEND_HANDLES = [
    mpatches.Patch(color=COLORS['baseline_a'], label='OpenAI CLIP (baseline)'),
    mpatches.Patch(color=COLORS['baseline_b'], label='BiomedCLIP (baseline)'),
    mpatches.Patch(color=COLORS['clip_std'],   label='CLIP — Standard captions'),
    mpatches.Patch(color=COLORS['clip_llm'],   label='CLIP — LLM captions'),
    mpatches.Patch(color=COLORS['bert_std'],   label='BERT — Standard captions'),
    mpatches.Patch(color=COLORS['bert_llm'],   label='BERT — LLM captions'),
]

# ── Group layout (10 groups, ordered: baselines | CLIP variants | BERT variants) ─
GROUPS = [
    # (x-label,        std_key,                llm_key,         color_std,      color_llm)
    ('OpenAI\nCLIP',   'OpenAI CLIP',           None,            'baseline_a',   None),
    ('Biomed\nCLIP',   'BiomedCLIP',             None,            'baseline_b',   None),
    ('CLIP\nMLP only', 'clip_mlp_std',           'clip_mlp_llm',  'clip_std',     'clip_llm'),
    ('CLIP\n+Image',   'clip_img_std',           'clip_img_llm',  'clip_std',     'clip_llm'),
    ('CLIP\n+Text',    'clip_txt_std',           'clip_txt_llm',  'clip_std',     'clip_llm'),
    ('CLIP\nFull FT',  'clip_full_std',          'clip_full_llm', 'clip_std',     'clip_llm'),
    ('BERT\nMLP only', 'bert_mlp_std',           'bert_mlp_llm',  'bert_std',     'bert_llm'),
    ('BERT\n+Image',   'bert_img_std',           'bert_img_llm',  'bert_std',     'bert_llm'),
    ('BERT\n+Text',    'bert_txt_std',           'bert_txt_llm',  'bert_std',     'bert_llm'),
    ('BERT\nFull FT',  'bert_full_std',          'bert_full_llm', 'bert_std',     'bert_llm'),
]

# ── Map raw model names to lookup keys ───────────────────────────────────────────
BUSI_KEYS = {
    'OpenAI CLIP':           'OpenAI CLIP',
    'BiomedCLIP':            'BiomedCLIP',
    'CLIP MLP Only (std)':   'clip_mlp_std',
    'CLIP +Img Enc (std)':   'clip_img_std',
    'CLIP +Txt Enc (std)':   'clip_txt_std',
    'CLIP Full (std)':       'clip_full_std',
    'CLIP MLP Only (LLM)':   'clip_mlp_llm',
    'CLIP +Img Enc (LLM)':   'clip_img_llm',
    'CLIP +Txt Enc (LLM)':   'clip_txt_llm',
    'CLIP Full (LLM)':       'clip_full_llm',
    'BERT MLP Only (std)':   'bert_mlp_std',
    'BERT +Img Enc (std)':   'bert_img_std',
    'BERT +Txt Enc (std)':   'bert_txt_std',
    'BERT Full (std)':       'bert_full_std',
    'BERT MLP Only (LLM)':   'bert_mlp_llm',
    'BERT +Img Enc (LLM)':   'bert_img_llm',
    'BERT +Txt Enc (LLM)':   'bert_txt_llm',
    'BERT Full (LLM)':       'bert_full_llm',
}

LIVER_KEYS = {
    'OpenAI CLIP':                                  'OpenAI CLIP',
    'BiomedCLIP':                                   'BiomedCLIP',
    'CLIP standard | MLP heads only':               'clip_mlp_std',
    'CLIP standard | MLP + image encoder':          'clip_img_std',
    'CLIP standard | MLP + text encoder':           'clip_txt_std',
    'CLIP standard | MLP + image + text encoder':   'clip_full_std',
    'CLIP LLM | MLP heads only':                    'clip_mlp_llm',
    'CLIP LLM | MLP + image encoder':               'clip_img_llm',
    'CLIP LLM | MLP + text encoder':                'clip_txt_llm',
    'CLIP LLM | MLP + image + text encoder':        'clip_full_llm',
    'BERT standard | MLP heads only':               'bert_mlp_std',
    'BERT standard | MLP + image encoder':          'bert_img_std',
    'BERT standard | MLP + text encoder':           'bert_txt_std',
    'BERT standard | MLP + image + text encoder':   'bert_full_std',
    'BERT LLM | MLP heads only':                    'bert_mlp_llm',
    'BERT LLM | MLP + image encoder':               'bert_img_llm',
    'BERT LLM | MLP + text encoder':                'bert_txt_llm',
    'BERT LLM | MLP + image + text encoder':        'bert_full_llm',
}


def load_values(csv_path, key_map):
    df = pd.read_csv(csv_path)
    return {key_map[row['model']]: row['accuracy']
            for _, row in df.iterrows() if row['model'] in key_map}


busi_vals  = load_values('BUSI_HO/classification_results.csv',         BUSI_KEYS)
liver_vals = load_values('Liver_HO/classification_results_final.csv',  LIVER_KEYS)


# ── Drawing function ─────────────────────────────────────────────────────────────

def draw_panel(ax, vals, panel_label):
    W = 0.32    # individual bar width
    GAP = 0.06  # gap between std and llm bars within a group
    xs = np.arange(len(GROUPS))

    for i, (xlabel, std_key, llm_key, c_std, c_llm) in enumerate(GROUPS):
        has_llm = llm_key is not None

        if has_llm:
            # Two bars: std shifted left, llm shifted right
            x_std = xs[i] - (W + GAP) / 2
            x_llm = xs[i] + (W + GAP) / 2
            if std_key in vals:
                ax.bar(x_std, vals[std_key], W, color=COLORS[c_std],
                       edgecolor='white', linewidth=0.5)
            if llm_key in vals:
                ax.bar(x_llm, vals[llm_key], W, color=COLORS[c_llm],
                       edgecolor='white', linewidth=0.5)
        else:
            # Single centered bar for baselines
            if std_key in vals:
                ax.bar(xs[i], vals[std_key], W, color=COLORS[c_std],
                       edgecolor='white', linewidth=0.5)

    # Vertical separators between the three model families
    for sep in [1.5, 5.5]:
        ax.axvline(sep, color='#cccccc', linewidth=1.0, linestyle='--', zorder=0)

    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in GROUPS], fontsize=8.5)
    ax.set_ylabel('Accuracy', fontsize=9)
    ax.set_xlabel('Model', fontsize=9)
    ax.set_xlim(-0.6, len(GROUPS) - 0.4)
    ax.set_ylim(0, 0.83)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.07, 1.04, panel_label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')


# ── Build figure ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

draw_panel(axes[0], busi_vals,  '(a) Breast')
draw_panel(axes[1], liver_vals, '(b) Liver')


fig.legend(handles=LEGEND_HANDLES, loc='lower center', ncol=3,
           fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.09))

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('figures_tables/zeroshot_grouped.pdf', bbox_inches='tight', dpi=200)
plt.savefig('figures_tables/zeroshot_grouped.png', bbox_inches='tight', dpi=200)
plt.close()
print('Saved figures_tables/zeroshot_grouped.pdf and .png')
