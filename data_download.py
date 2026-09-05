import os
import requests
import pandas as pd

def get_latest_10k_filing_url(cik="0001652044"):
    """
    Gets the URL of the latest 10-K filing for a given CIK
    using the SEC's more reliable JSON data feed.
    """
    print("Fetching latest 10-K filing URL from the SEC JSON API...")
    padded_cik = cik.zfill(10)
    json_url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
    headers = {'User-Agent': 'Gemini Financial Analysis Project student@example.com'}
    
    try:
        response = requests.get(json_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        recent_filings = data['filings']['recent']
        for i in range(len(recent_filings['form'])):
            if recent_filings['form'][i] == '10-K':
                accession_number = recent_filings['accessionNumber'][i].replace('-', '')
                primary_document = recent_filings['primaryDocument'][i]
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{primary_document}"
                print(f"Successfully found 10-K URL: {filing_url}")
                return filing_url
        print("Error: Could not find a 10-K filing in the recent filings.")
        return None
    except Exception as e:
        print(f"Error fetching or parsing SEC JSON data: {e}")
        return None

def download_and_save_raw_html(url, output_path="data/alphabet_10k_submission.txt"):
    """
    Downloads the raw HTML content from the URL and saves it to a text file,
    matching the article's methodology.
    """
    if not url:
        print("Download skipped as no URL was provided.")
        return False
        
    print(f"Downloading raw HTML from: {url}")
    headers = {'User-Agent': 'Gemini Financial Analysis Project student@example.com'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # --- KEY CHANGE ---
        # We save the raw response text (HTML) instead of parsed text.
        # This is what `unstructured` needs to identify element types.
        html_content = response.text
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"SUCCESS: Raw HTML submission saved to '{output_path}'")
        return True
    except Exception as e:
        print(f"An error occurred during raw HTML download: {e}")
        return False

def create_structured_data(output_path="data/alphabet_financials_structured.csv"):
    """Creates the structured tabular CSV file."""
    # This function remains the same as before.
    print("\nCreating structured tabular data...")
    revenue_data = {
        'year': [2023, 2023, 2023, 2023, 2022, 2022, 2022, 2022],
        'quarter': ['Q4', 'Q3', 'Q2', 'Q1', 'Q4', 'Q3', 'Q2', 'Q1'],
        'revenue_usd_billions': [86.31, 76.69, 74.60, 69.79, 76.05, 69.09, 69.69, 68.01],
        'net_income_usd_billions': [20.69, 19.69, 18.37, 15.05, 13.62, 13.91, 16.00, 16.44]
    }
    df = pd.DataFrame(revenue_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"SUCCESS: Structured tabular data has been saved to '{output_path}'")
    return True

if __name__ == "__main__":
    # Rerun the data preparation. This will now save the correct files.
    filing_url = get_latest_10k_filing_url()
    download_and_save_raw_html(filing_url)
    create_structured_data()
    print("\nAll data has been successfully prepared.")