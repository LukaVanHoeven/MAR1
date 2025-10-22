from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


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

def mean_ci(data, confidence=0.95):
    data = pd.to_numeric(data, errors='coerce').dropna()
    n = len(data)
    m = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, se, h

graphical_mean, graphical_se, graphical_ci = mean_ci(graphical)
terminal_mean, terminal_se, terminal_ci = mean_ci(terminal)

print("\n--- Descriptive Statistics (APA 7) ---")
print(f"Graphical chat: M = {graphical_mean:.2f}, SD = {graphical.std(ddof=1):.2f}, "
      f"95% CI [{graphical_mean - graphical_ci:.2f}, {graphical_mean + graphical_ci:.2f}]")
print(f"Terminal chat:  M = {terminal_mean:.2f}, SD = {terminal.std(ddof=1):.2f}, "
      f"95% CI [{terminal_mean - terminal_ci:.2f}, {terminal_mean + terminal_ci:.2f}]")

t_value, p_value = stats.ttest_rel(graphical, terminal)
df_t = len(graphical) - 1
diff = graphical.to_numpy() - terminal.to_numpy()
cohen_d = diff.mean() / diff.std(ddof=1)

print("\n--- Inferential Statistics ---")
print(f"t({df_t}) = {t_value:.2f}, p = {p_value:.4f}, Cohen’s d = {cohen_d:.2f}")
if p_value < 0.05:
    print("→ Statistically significant difference between chat types.")
else:
    print("→ No statistically significant difference found.")

# Bar plot 
means = [graphical_mean, terminal_mean]
errors = [graphical_se * stats.t.ppf(0.975, len(graphical)-1),
          terminal_se * stats.t.ppf(0.975, len(terminal)-1)]

plt.figure(figsize=(7,5))
bars = plt.bar(["Graphical chat", "Terminal chat"], means, yerr=errors, capsize=6,
               color=["#4C72B0", "#55A868"], alpha=0.8)
plt.ylabel("Mean SUS Score (0–100)", fontsize=12)
plt.title("Mean System Usability Scale (SUS) Scores by Chat Type", fontsize=14, fontweight="bold")
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.6)
for bar, mean in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width()/2, mean + 7,
             f"{mean:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.show()

# Paired line plot (participant-level comparison)
plt.figure(figsize=(7,5))
for g, t in zip(graphical, terminal):
    plt.plot(["Graphical chat", "Terminal chat"], [g, t], color="gray", alpha=0.4, linewidth=1)
plt.scatter(["Graphical chat"]*len(graphical), graphical, color="#4C72B0", s=60)
plt.scatter(["Terminal chat"]*len(terminal), terminal, color="#55A868", s=60)
plt.ylabel("SUS Score (0–100)", fontsize=12)
plt.title("Participant-level SUS Comparison", fontsize=14, fontweight="bold")
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.text(0.5, 95, f"p = {p_value:.4f}", ha="center", fontsize=12)
plt.tight_layout()
plt.show()

# Violin plot
plt.figure(figsize=(8,6))
sns.violinplot(x="Chat_Type", y="SUS_Score", data=new_df,
               inner=None, palette=["#4C72B0", "#55A868"], alpha=0.6)
sns.swarmplot(x="Chat_Type", y="SUS_Score", data=new_df,
              color="black", size=5, alpha=0.7)
plt.title("Distribution of SUS Scores by Chat Type", fontsize=14, fontweight="bold")
plt.ylabel("SUS Score (0–100)", fontsize=12)
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.text(0.5, 95, f"p = {p_value:.4f}", ha="center", fontsize=12)
plt.tight_layout()
plt.show()
