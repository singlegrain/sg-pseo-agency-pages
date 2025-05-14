import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os
from urllib.parse import urlparse, unquote

# Input and output paths
CSV_PATH = os.path.join(os.path.dirname(__file__), '../input/pages.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../output/keywords_posts.py')

# Helper to extract post ID from <body> class
def extract_post_id(html):
    soup = BeautifulSoup(html, 'lxml')
    body = soup.find('body')
    if not body or not body.has_attr('class'):
        return None
    for cls in body['class']:
        m = re.match(r'(?:postid|page-id)-(\d+)', cls)
        if m:
            return m.group(1)
    return None

# Helper to generate keyword from URL
def extract_keyword(url):
    path = urlparse(url).path
    path = path.strip('/')
    segments = [seg for seg in path.split('/') if seg]
    if not segments:
        return None
    slug = segments[-1]
    keyword = unquote(slug.replace('-', ' ').replace('_', ' '))
    # Only append ' agency' if the keyword is in English (does not contain 'agencia')
    if 'agency' in path and 'agency' not in keyword and 'agencia' not in keyword.lower():
        keyword += ' agency'
    return keyword

# Read CSV and filter
pages = pd.read_csv(CSV_PATH)
filtered = pages[pages['Clicks'] >= 2]

results = []
for url in filtered['Top pages']:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        post_id = extract_post_id(resp.text)
        if not post_id:
            print(f"[WARN] No post ID found for {url}")
            continue
        keyword = extract_keyword(url)
        if not keyword:
            print(f"[WARN] No keyword found for {url}")
            continue
        results.append({
            'keyword': keyword,
            'post_id': post_id
        })
        print(f"[OK] {url} -> post_id={post_id}, keyword='{keyword}'")
    except Exception as e:
        print(f"[ERROR] Failed to process {url}: {e}")

# Write output as a Python constant
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write('KEYWORDS_POSTS = [\n')
    for item in results:
        f.write(f"    {{'keyword': '{item['keyword']}', 'post_id': '{item['post_id']}'}} ,\n")
    f.write(']\n')

print(f"\nSaved {len(results)} entries to {OUTPUT_PATH}") 