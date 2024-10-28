import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.robotparser import RobotFileParser
import random
import numpy as np

# User-agent rotation
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:85.0) Gecko/20100101 Firefox/85.0',
]

def check_robots_txt(url):
    rp = RobotFileParser()
    rp.set_url(url + "/robots.txt")
    rp.read()
    return rp

def get_links_from_sgtimes(category_sgt, headers, num_limit, rp):
    full_url = "https://thesaigontimes.vn" + category_sgt

    if not rp.can_fetch(headers['User-Agent'], full_url):
        return []

    response = requests.get(full_url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')

    links = soup.find_all(
        'h3', class_='entry-title td-module-title', limit=num_limit)
    href_link = []
    for link in links:
        try:
            href_link.append(link.find('a')['href'])
            time.sleep(1)
        except:
            pass

    return list(set(href_link))

def get_links_from_cnn(category, headers, num_limit, rp):
    full_url = "https://edition.cnn.com" + category

    if not rp.can_fetch(headers['User-Agent'], full_url):
        return []

    response = requests.get(full_url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')

    links = set()
    page_number = 1

    while len(links) < num_limit:
        response = requests.get(full_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        for item in soup.find_all('a', class_='container__link'):
            links.add("https://edition.cnn.com" + item['href'])
            if len(links) >= num_limit:
                break

        page_number += 1
        time.sleep(1)  # Throttle requests

    return list(links)[:num_limit]

def get_links_from_bbc(category, headers, num_limit, rp):
    full_url = "https://www.bbc.com" + category

    if not rp.can_fetch(headers['User-Agent'], full_url):
        return []

    response = requests.get(full_url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')

    links = soup.find_all('a', {'data-testid':'internal-link'}, limit=num_limit)
    href_link = []
    for link in links:
        try:
            href_link.append("https://www.bbc.com" + link['href'])
            time.sleep(1)
        except:
            pass

    return list(set(href_link))

def get_text_from_link(url, headers, failed_links):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            failed_links.append(url)
            return np.nan
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = ' '.join(paragraph.get_text(strip=True) for paragraph in paragraphs)
        return text_content
    except Exception as e:
        failed_links.append(url)
        return np.nan

def get_texts_from_links_parallel(links, headers):
    failed_links = []
    results_dict = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(
            get_text_from_link, url, headers, failed_links): url for url in links}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                results_dict[url] = result
                # Random delay to throttle requests
                time.sleep(random.uniform(1, 3))
            except Exception as exc:
                failed_links.append(url)
                results_dict[url] = np.nan
    results = [results_dict[url] for url in links]
    return results, failed_links

def run_parallel_processing(links, headers):
    texts, failed_links = get_texts_from_links_parallel(links, headers)
    df = pd.DataFrame({'url': links, 'txt': texts})
    return df, failed_links

if __name__ == "__main__":
    category_cnn = '/business'  # Example category for CNN
    category_sgt = '/tai-chinh-ngan-hang'  # Example category for Saigon Times
    category_bbc = '/innovation'  # Example category for BBC

    headers = {
        'User-Agent': random.choice(user_agents)
    }
    num_limit = 10

    # Check robots.txt
    cnn_rp = check_robots_txt("https://edition.cnn.com")
    sgt_rp = check_robots_txt("https://thesaigontimes.vn")
    bbc_rp = check_robots_txt("https://www.bbc.com")

    links_cnn = get_links_from_cnn(category_cnn, headers, num_limit, cnn_rp)
    links_sgt = get_links_from_sgtimes(category_sgt, headers, num_limit, sgt_rp)
    links_bbc = get_links_from_bbc(category_bbc, headers, num_limit, bbc_rp)

    df_txt, failed_links = run_parallel_processing(links_sgt+links_cnn+links_bbc, headers)
    df_txt.to_csv('scraped_texts_check.csv', index=False)
