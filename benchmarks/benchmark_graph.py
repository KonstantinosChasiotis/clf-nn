import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_performance(csv_file, title, out_name):
   # Load
   df = pd.read_csv(csv_file)
   channels = df["Channels"].astype(int).values
   perf = df["Flops"].values / df["Cycles"].values


   # Compute positions so points are equally spaced
   pos = np.arange(len(channels))


   # Set up the figure
   fig, ax = plt.subplots(figsize=(10, 6))
   fig.patch.set_facecolor("white")
   ax.set_facecolor("white")
   ax.grid(color="lightgray", linestyle="-", linewidth=1, alpha=0.8)


   # Plot
   ax.plot(pos, perf, marker="o", linestyle="-", color="blue", linewidth=1.5)


   # Equally spaced x-axis ticks, labeled by channel count
   ax.set_xticks(pos)
   ax.set_xticklabels([str(c) for c in channels], rotation=45)
   ax.set_xlabel("Channels", fontsize=12, labelpad=10)


   # Title and annotation
   ax.set_title(
       title,
       fontsize=12,
       fontweight="bold",
       fontname="Calibri",
       loc="left",
       pad=20
   )
   ax.text(
       ax.get_xlim()[0],
       ax.get_ylim()[1] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
       "[FLOPs/cycle]",
       fontsize=12,
       ha="left",
       va="center",
       fontname="Helvetica"
   )


   # Save & show
   plt.subplots_adjust(bottom=0.15, right=0.82)
   plt.savefig(out_name, dpi=300,
               bbox_inches="tight",
               facecolor=fig.patch.get_facecolor())
   plt.show()




# Linear
plot_performance(
   "baseline_perfplot_linear.csv",
   "CliffordLinear Performance",
   "linear_performance_baseline.png"
)


# GroupNorm
plot_performance(
   "baseline_perfplot_groupnorm.csv",
   "CliffordGroupNorm Performance",
   "groupnorm_performance_baseline.png"
)


