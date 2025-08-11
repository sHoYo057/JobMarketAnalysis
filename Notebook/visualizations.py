import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def top_skills_frequency(df, min_count=5):
    skill_counts = {}
    for skills_list in df["skills"]:
        for skill in skills_list:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
    freq_df = pd.DataFrame(skill_counts.items(), columns=["Skill", "Count"])
    freq_df = freq_df[freq_df["Count"] >= min_count].sort_values(by="Count", ascending=False)
    freq_df.reset_index(drop=True, inplace=True)
    return freq_df

def plot_top_skills_bar(freq_df, top_n=15):
    plt.figure(figsize=(12,6))
    sns.barplot(data=freq_df.head(top_n), x='Count', y='Skill', palette='viridis')
    plt.title(f"Top {top_n} Skills in Job Postings")
    plt.xlabel("Number of Job Listings")
    plt.ylabel("Skill")
    plt.tight_layout()
    plt.show()

def plot_skills_heatmap(df, skills_list):
    locations = df['location'].unique()
    heatmap_data = pd.DataFrame(0, index=skills_list, columns=locations)
    for idx, row in df.iterrows():
        loc = row['location']
        for skill in row['skills']:
            if skill in skills_list:
                heatmap_data.loc[skill, loc] += 1
    plt.figure(figsize=(14,8))
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu')
    plt.title("Skill Demand Heatmap by Location")
    plt.xlabel("Location")
    plt.ylabel("Skill")
    plt.tight_layout()
    plt.show()
