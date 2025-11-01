#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import sys


def load_sections(csv_file):
    """
    Parses a CSV with sections of the form:
      FunctionName
      batch,cycles,flops
      16,67042.96,132096
      ...
    Returns a dict mapping FunctionName -> DataFrame.
    """
    sections = {}
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if ',' not in line:
            func = line
            i += 1
            header = lines[i].strip()
            i += 1
            buf = [header + '\n']
            while i < len(lines) and lines[i].strip() and ',' in lines[i]:
                buf.append(lines[i])
                i += 1
            df = pd.read_csv(io.StringIO(''.join(buf)))
            sections[func] = df
        else:
            i += 1
    return sections


def plot_performance(csv_file, title, out_name):
    data = load_sections(csv_file)

    # Create 3:2 landscape canvas
    fig, ax = plt.subplots(figsize=(12, 8))
    # Light grey plot background only
    ax.set_facecolor('lightgrey')
    # Thicken spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    # Tick styling: larger, bold
    ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(14)
        lbl.set_fontweight('bold')

    # Original grid styles
    ax.xaxis.grid(color='white', linestyle='-', linewidth=1, alpha=0.0)
    ax.yaxis.grid(color='white', linestyle='-', linewidth=2, alpha=0.8)

        # Plot each function's performance
    for func, df in data.items():
        df = df.sort_values('batch')
        df['log2_batch'] = np.log2(df['batch'])
        perf = df['flops'] / df['cycles']
        ax.plot(
            df['log2_batch'], perf,
            marker='o', linestyle='-', linewidth=4, markersize=8,
            label=func
        )
    # Set x-ticks only
    unique = sorted({np.log2(b) for df in data.values() for b in df['batch']})
    ax.set_xticks(unique)

        # Three-line left-aligned title
    left_x = 0.01
    fig.text(left_x, 0.98, title,
             ha='left', va='top', fontsize=17, fontweight='bold')
    fig.text(left_x, 0.945, 'Intel core i5-1035g4 @ 3.70GHz',
             ha='left', va='top', fontsize=17, fontweight='bold')
    fig.text(left_x, 0.91, 'Performance [FLOPs/cycle] vs. Batch size',
             ha='left', va='top', fontsize=15)

    # Legend upper left with transparent background
    ax.legend(loc='upper right', prop={'size': 12, 'weight': 'bold'}, framealpha=0)

    # Tight layout and save
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_name, dpi=300)
    print(f"Saved styled performance plot to {out_name}")
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <csv_file> <plot_title> <out_png>")
        sys.exit(1)
    _, csv_file, title, out_png = sys.argv
    plot_performance(csv_file, title, out_png)
