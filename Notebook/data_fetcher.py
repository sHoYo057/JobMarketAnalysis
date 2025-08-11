import requests
import pandas as pd
from skill_extractor import extract_skills

APP_ID = "2e5ae062"
APP_KEY = "808ef5101eab10c5608449c29dcdf259"

def fetch_adzuna_jobs(keyword, country="us", pages=1, results_per_page=50):
    all_jobs = []

    for page in range(1, pages + 1):
        endpoint = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
        params = {
            "app_id": APP_ID,
            "app_key": APP_KEY,
            "results_per_page": results_per_page,
            "what": keyword,
            "content-type": "application/json"
        }
        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            data = response.json()
            all_jobs.extend(data.get("results", []))
        else:
            print(f"Error fetching page {page}: {response.status_code}")
            break

    cleaned_jobs = []
    for job in all_jobs:
        description = job.get("description", "").strip()
        cleaned_jobs.append({
            "job_title": job.get("title", "").strip(),
            "company": job.get("company", {}).get("display_name", "").strip(),
            "location": job.get("location", {}).get("display_name", "").strip(),
            "salary_min": job.get("salary_min", None),
            "salary_max": job.get("salary_max", None),
            "category": job.get("category", {}).get("label", "").strip(),
            "description": description,
            "contract_type": job.get("contract_type", "").strip(),
            "contract_time": job.get("contract_time", "").strip(),
            "created": job.get("created", ""),
            "redirect_url": job.get("redirect_url", ""),
            "skills": extract_skills(description)
        })

    df = pd.DataFrame(cleaned_jobs)
    df.drop_duplicates(subset=["job_title", "company", "location", "description"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
