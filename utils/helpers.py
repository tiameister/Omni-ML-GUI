import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from utils.text import normalize_text as _normalize_text
from utils.text import normalize_quotes_ascii as _normalize_quotes_ascii

def normalize_text(s: str) -> str:
    return _normalize_text(s)

def ensure_outdir(p="output"):
    os.makedirs(p, exist_ok=True)
    return p

# Normalize fancy quotes/backticks and the Unicode replacement char to ASCII
def normalize_quotes_ascii(s: str) -> str:
    return _normalize_quotes_ascii(s)

def save_bar(fig_path: str, labels: List[str], values: List[float], title: str, xlabel: str):
    sns.set_style('whitegrid')
    fig_h = max(3.0, 0.5 * len(labels))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ypos = np.arange(len(labels))
    bars = ax.barh(ypos, values, color='#64B5F6', edgecolor='#1E88E5', alpha=0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.7, alpha=0.5)
    # Annotate values at end of bars
    xlim = ax.get_xlim()
    span = xlim[1] - xlim[0]
    for y, b, v in zip(ypos, bars, values):
        x = b.get_width()
        off = span * 0.01
        ha = 'left' if x >= 0 else 'right'
        ax.text(x + (off if x >= 0 else -off), y, f"{v:.3f}", va='center', ha=ha, fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
