from data_fetcher import fetch_adzuna_jobs
from visualizations import top_skills_frequency, plot_top_skills_bar, plot_skills_heatmap
from slide_deck import create_slide_deck
from nlp_tools import get_top_tfidf_keywords, bert_cluster_job_descriptions

def main():
    # Fetch jobs
    jobs_df = fetch_adzuna_jobs("data scientist", country="us", pages=2)
    print(f"Fetched {len(jobs_df)} jobs.")

    # Skills frequency
    freq_df = top_skills_frequency(jobs_df)
    print(freq_df.head())

    # Visualizations
    plot_top_skills_bar(freq_df)
    plot_skills_heatmap(jobs_df, freq_df['Skill'].head(7).tolist())

    # NLP: TF-IDF keywords
    tfidf_keywords = get_top_tfidf_keywords(jobs_df["description"].tolist())
    print("Top TF-IDF keywords:", tfidf_keywords)

    # BERT clustering example
    clusters, _ = bert_cluster_job_descriptions(jobs_df["description"].tolist(), n_clusters=4)
    jobs_df['cluster'] = clusters
    print(jobs_df[['job_title', 'cluster']].head())

    # Save slide deck
    create_slide_deck(freq_df)

if __name__ == "__main__":
    main()