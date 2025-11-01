#!/usr/bin/env python3
# === Testbench CPU Configuration ===
# Parameter | Value
# Processor manufacturer, name, number, microarchitecture | Intel Core i5-1035G4, Ice Lake
# CPU Base Frequency | 1.10 GHz (nominal)
# CPU Maximum Frequency, Frequency Scaling support | 3.70 GHz, supports Turbo Boost
# Phase in Intel's development model | Tick (Ice Lake is a process shrink)

# Datasheet: https://www.intel.com/content/www/us/en/products/sku/196591/intel-core-i51035g4-processor-6m-cache-up-to-3-70-ghz/specifications.html
# Max Memory Bandwidth: 58.3 GB/s => 53 bytes/cycle at 1.10GHz
# 4 cores, 8 threads. Supports AVX2/AVX-512

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO
import argparse
import os

class RooflinePlot:
    def _load_data(self, csv_path: str):
        try:
            content = open(csv_path, 'r').read()
        except Exception as e:
            print(f"Error reading file {csv_path}: {e}")
            return None
        sections, current, buffer = {}, None, []
        for line in content.splitlines():
            s = line.strip()
            if not s:
                continue
            if ',' not in s and not s.lower().startswith('batch,cycles'):
                if current and buffer:
                    sections[current] = "\n".join(buffer)
                current, buffer = s, []
            elif s.lower().startswith('batch,cycles'):
                if not current:
                    current = 'Data'
                buffer.append(s)
            elif current:
                buffer.append(s)
        if current and buffer:
            sections[current] = "\n".join(buffer)
        data = {}
        for name, txt in sections.items():
            df = pd.read_csv(StringIO(txt), skipinitialspace=True)
            df.columns = df.columns.str.strip()
            if df.empty:
                continue
            for c in ['flops','bytes_read','cycles']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df.dropna(subset=['flops','bytes_read','cycles'], inplace=True)
            df['oi'] = df['flops'] / df['bytes_read'].replace({0:np.nan})
            df['perf'] = df['flops'] / df['cycles'].replace({0:np.nan})
            valid = df[(df['oi']>0) & (df['perf']>0)]
            if not valid.empty:
                data[name] = list(zip(valid['oi'], valid['perf']))
        return data or None

    def plot(self, csv_path: str, save_path: str = None):
        data = self._load_data(csv_path)
        peak, avx2, bw = 4, 32, 53
        # 3:2 landscape canvas
        fig, ax = plt.subplots(figsize=(12, 8))
        # Thicken axis spines and ticks
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontsize(14)
            lbl.set_fontweight('bold')
        # Light grey plot background only
        ax.set_facecolor('lightgrey')
        # Log scales and minor ticks
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.minorticks_on()
        x = np.logspace(-3, 3, 200)
        # Rooflines: thicker lines
        ax.plot(x, bw * x, '-', lw=4, color='blue')
        ax.axhline(peak, ls='--', lw=4, color='red')
        ax.axhline(avx2, ls='--', lw=4, color='green')

        # Data series
        if data:
            for name, pts in data.items():
                xi, yi = zip(*pts)
                ax.plot(xi, yi, 'o', markersize=6, label=name)

        # Fixed ranges
        ax.set_xlim(1e-3, 1e3)
        ax.set_ylim(1e-1, 1e2)

        # Annotate Memory Roof: horizontal label at bottom-left of plot box
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x0 = x_min * 1.2
        y0 = y_min * 1.2
        ax.text(
            x0, y0,
            f'Memory Roof (β) = {bw} B/cycle',
            color='blue', rotation=0,
            ha='left', va='bottom',
            fontsize=16, fontweight='bold'
        )

        # Annotate peaks inside plot and pushed right
        xp = x_max * 0.8
        ax.text(xp, peak * 1.1, f'Scalar Peak (π) = {peak} FLOPS/cycle',
                color='red', ha='right', va='bottom', fontsize=16, fontweight='bold')
        ax.text(xp, avx2 * 1.1, f'AVX2 Peak (π) = {avx2} FLOPS/cycle',
                color='green', ha='right', va='bottom', fontsize=16, fontweight='bold')

        # Left-aligned three-line title with specified font sizes
        fig.suptitle('Intel Core i5-1035G4 @ 3.70GHz', x=0.01, y=0.98,
                     ha='left', va='top', fontsize=17, fontweight='bold')
        fig.text(0.01, 0.945, 'Compiler: gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0',
                 ha='left', va='top', fontsize=15)
        fig.text(0.01, 0.91, 'Performance (F/C) vs Operational Intensity (F/B)',
                 ha='left', va='top', fontsize=13)

        # White grid lines, thicker, both major and minor
        ax.grid(True, which='major', color='white', linestyle='-', linewidth=2)
        ax.grid(True, which='minor', color='white', linestyle='-', linewidth=1)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='lower right', prop={'size': 11, 'weight': 'bold'})

        # Tight layout
        fig.tight_layout(rect=[0, 0, 1, 0.85])

        # Save
        if not save_path:
            base = os.path.splitext(os.path.basename(csv_path))[0]
            save_path = f"{base}_roofline.png"
        fig.savefig(save_path, dpi=300)
        print(f"Saved roofline plot to {save_path}")
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path')
    parser.add_argument('--out', '-o', help='Output image path', default=None)
    args = parser.parse_args()
    RooflinePlot().plot(args.csv_path, save_path=args.out)
