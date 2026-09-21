"""Draw recorded aggregate findings; this does not rerun research models."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/research-overview.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig = plt.figure(figsize=(15, 7.6), facecolor="#091c2e")
fig.text(.055, .91, "MARITIME FUSION OPS", color="#64d8d4", size=13, weight="bold")
fig.text(.055, .83, "From thermal observations to evidence review", color="white", size=25, weight="bold")
fig.text(.055, .775, "Recorded research · September 2025–February 2026 · Three study regions", color="#bacddd", size=12)

left = fig.add_axes([.055, .25, .27, .43], facecolor="#102b42")
left.axis("off")
left.text(.06, .90, "DATA FOUNDATION", color="#64d8d4", weight="bold", transform=left.transAxes)
for y, count, label in [( .69, "11,525", "clean satellite detections"), (.43, "3,589", "multilingual articles"), (.17, "3,289", "candidate thermal events")]:
    left.text(.06, y, count, color="white", size=25, weight="bold", transform=left.transAxes)
    left.text(.06, y-.095, label, color="#bacddd", size=11, transform=left.transAxes)

mid = fig.add_axes([.36, .33, .28, .31], facecolor="#091c2e")
mid.barh([1, 0], [.312, .284], color=["#64d8d4", "#7e9ebd"], height=.45)
mid.set_yticks([1, 0], ["Geography RF", "Satellite +\ncontext RF"], color="white", size=10)
mid.set_xlim(0, .4)
mid.set_xticks([0, .1, .2, .3, .4])
mid.tick_params(axis="x", colors="#bacddd")
mid.tick_params(axis="y", length=0)
mid.set_xlabel("Pooled average precision", color="#bacddd")
mid.set_title("FIVE-MONTH COMPARISON", color="#64d8d4", loc="left", size=11, weight="bold", pad=30)
mid.spines[["top", "right", "left", "bottom"]].set_visible(False)
for y, value in [(1, .312), (0, .284)]:
    mid.text(value+.009, y, f"{value:.3f}", color="white", va="center", weight="bold")

right = fig.add_axes([.71, .25, .255, .43], facecolor="#102b42")
right.axis("off")
right.text(.04, .90, "SEVEN CASES REVIEWED", color="#64d8d4", weight="bold", transform=right.transAxes)
for y, count, label, color in [(.68,"3","Ordinary context compatible", "#8bb3d9"),(.42,"4","Unexplained anomalies", "#ffb56b"),(.16,"0","Causally confirmed conflict", "white")]:
    right.text(.04,y,count,color=color,size=28,weight="bold",transform=right.transAxes)
    right.text(.17,y+.025,label,color="#dce8f2",size=10,transform=right.transAxes)

fig.text(.055, .13, "Analyst-support prototype. News association is not confirmed conflict.", color="white", size=13, weight="bold")
fig.text(.055, .077, "Historical metrics are exploratory. The live collector separately uses confidence / FRP rules.", color="#bacddd", size=11)
fig.savefig(OUT, dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)
print(OUT)
