from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

file_path = r"C:\Users\HP\Downloads\MAIR 2 user test.csv (1)\MAIR 2 user test.csv"
df = pd.read_csv(file_path)

columns = []
SUS_1 = []  
SUS_2 = [] 

for column in df.columns:
    if column.startswith("I "):  
        columns.append(column)

for column in columns:
    if column.endswith(".1"):
        SUS_2.append(column)
    else:
        SUS_1.append(column)

def compute_sus_score(values):
    score = 0
    for i, v in enumerate(values, start=1):
        if i % 2 == 1: 
            score += v - 1
        else:       
            score += 5 - v
    return score * 2.5

SUS_1_scores = []
SUS_2_scores = []

for _, row in df.iterrows():
    SUS_1_scores.append(compute_sus_score(row[SUS_1]))
    SUS_2_scores.append(compute_sus_score(row[SUS_2]))

df["SUS_1"] = SUS_1_scores
df["SUS_2"] = SUS_2_scores

sus_scores_bychat = []

for i, row in df.iterrows():
    used_chat = row["Which chat did you use?"].strip().lower()
    participant = i + 1

    sus_scores_bychat.append({
        "Participant": participant,
        "Chat_Type": "Graphical chat" if "graphical" in used_chat else "Terminal chat",
        "SUS_Score": row["SUS_1"]
    })

    sus_scores_bychat.append({
        "Participant": participant,
        "Chat_Type": "Terminal chat" if "graphical" in used_chat else "Graphical chat",
        "SUS_Score": row["SUS_2"]
    })

new_df = pd.DataFrame(sus_scores_bychat)

graphical = new_df[new_df["Chat_Type"].str.contains("Graphical", case=False)]["SUS_Score"]
terminal = new_df[new_df["Chat_Type"].str.contains("Terminal", case=False)]["SUS_Score"]

t_value, p_value = stats.ttest_rel(graphical, terminal)
print(f"Graphical mean = {graphical.mean():.2f}")
print(f"Terminal mean  = {terminal.mean():.2f}")
print(f"t_value = {t_value:.3f}, p_value = {p_value:.4f}")

#Graphical/terminal chat bar plot
means = [graphical.mean(), terminal.mean()]
errors = [graphical.std(ddof=1)/np.sqrt(len(graphical)),
          terminal.std(ddof=1)/np.sqrt(len(terminal))]
plt.figure(figsize=(7,5))
bars = plt.bar(["Graphical chat", "Terminal chat"], means,
               yerr=errors, capsize=6,
               color=["#4C72B0", "#55A868"], alpha=0.8)
plt.ylabel("Average SUS Score")
plt.title("Average SUS Score by Chat Type")
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.6)
for bar, mean in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width()/2, mean + 3,
             f"{mean:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.show()

#Paired line plot
plt.figure(figsize=(7,5))
for g, t in zip(graphical, terminal):
    plt.plot(["Graphical chat", "Terminal chat"], [g, t],
             color="gray", alpha=0.5, linewidth=1)
plt.scatter(["Graphical chat"]*len(graphical), graphical, color="#4C72B0", s=60, label="Graphical")
plt.scatter(["Terminal chat"]*len(terminal), terminal, color="#55A868", s=60, label="Terminal")
plt.ylabel("SUS Score")
plt.title("SUS Comparison per Participant")
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.text(0.5, 95, f"p = {p_value:.4f}", ha="center", fontsize=12)
plt.tight_layout()
plt.show()

#Boxplot
plt.figure(figsize=(7,5))
plt.boxplot(
    [graphical, terminal],
    labels=["Graphical chat", "Terminal chat"],
    patch_artist=True,
    boxprops=dict(facecolor="#4C72B0", alpha=0.6),
    medianprops=dict(color="black", linewidth=2)
)
plt.ylabel("SUS Score")
plt.title("Distribution of SUS Scores by Chat Type")
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.text(1.5, 95, f"p = {p_value:.4f}", ha="center", fontsize=12)
plt.tight_layout()
plt.show()
