# Job Market Analysis

This project analyzes the job market by fetching job postings from the Adzuna API and extracting insights such as top demanded skills, keyword trends, and semantic clusters using NLP techniques. It includes a Streamlit app for interactive exploration.

## Features

- Fetch fresh job data by keyword and country
- Extract and visualize top demanded skills
- Perform TF-IDF keyword extraction on job descriptions
- Cluster job descriptions semantically using BERT embeddings
- Interactive dashboard with skill heatmaps and clustering results
- Generate downloadable PowerPoint reports

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sHoYo057/JobMarketAnalysis.git
   cd JobMarketAnalysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
    ```bash
   streamlit run app_enhanced.py
   
   ```
## Configuration
Update your Adzuna API credentials (APP_ID and APP_KEY) in the app code before running.

## Notes
The Adzuna API may have rate limits; adjust fetch parameters accordingly.

Expand the skills list in the code to customize skill extraction.




