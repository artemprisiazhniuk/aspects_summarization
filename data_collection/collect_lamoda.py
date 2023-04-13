import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import re
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def get_links(URL, links_limit=None, items_per_page=60):
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36'}

    links = []
    ids = []

    # URL = "https://www.lamoda.ru/c/355/clothes-zhenskaya-odezhda/?page={}"
    page = requests.get(URL.format(1), headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")
    num_pages = soup.find('span', class_='ui-catalog-search-head-products-count')
    if not num_pages: 
        num_pages = 3000 # grows every day
    else:
        num_pages = int(''.join(re.findall(r"[0-9]+", num_pages.text)))
    if links_limit:
        num_pages = min(num_pages, links_limit // items_per_page)

    for i in tqdm(range(1, num_pages)):
        page = requests.get(URL.format(i), headers=headers)
        soup = BeautifulSoup(page.content, "html.parser")

        for item in soup.findAll('div', class_='x-product-card__card'):
            a = item.find('a')
            if a.has_attr('href'):
                a = a['href']
                sku = a.split('/')[2]
                a = 'https://www.lamoda.ru' + a
                
                links.append(a)
                ids.append(sku)

    links = list(set(links))
    ids = list(set(ids))

    return links, ids


def get_reviews(links, ids, items_limit=None, reviews_limit=None, name='lamoda'):
    scheme = "https://www.lamoda.ru/api/v1/product/reviews?limit=50&offset={}&sku={}&sort=date&sort_direction=desc&only_with_photos=false"

    tokenize = lambda x: re.findall(r"[\w']+|[.,!?;]+", x)

    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)

    with open(f'../data/data_clothes_{name}.jsonl', 'w+', encoding='utf-8') as f:
        f.write('[\n')

        if items_limit: links = links[:items_limit]
        for i in tqdm(range(len(links))):
            url = links[i]
            id = ids[i].upper()

            headers = {
                "Cookie": "is_seo_or_robot=seo; sid=MzliODZkNjVmMGI0MmNmZTliNzY5ODVhMjAwODY1NDY=|1673185947|703b76e41be1ea8fb5de67c1230f53c52582109f",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
            }

            reviews = requests.get(scheme.format(0, id), headers=headers)
            reviews_dict = json.loads(reviews.text)
            if 'reviews' not in reviews_dict: continue
            num_reviews = len(reviews_dict['reviews'])
            if num_reviews is None: continue
            num_reviews = min(num_reviews, reviews_limit)

            for j in range(0, num_reviews, 50):
                reviews = requests.get(scheme.format(j, id), headers=headers)

                if reviews.text:
                    reviews_dict = json.loads(reviews.text)

                    new_item = {'id': id, 'reviews': []}

                    for review in reviews_dict['reviews']:
                        review_rating = float(review['rating'])

                        review_text = review['text']
                        review_text = emoji_pattern.sub(r'', review_text)
                        if not review_text: continue # skip if empty review

                        review_sentences = [' '.join(tokenize(t)) for t in sent_tokenize(review_text, language='russian')]

                        new_review = {'sentences': review_sentences, 'rating': review_rating}
                        new_item['reviews'].append(new_review)

                    if len(new_item['reviews']) <= 0: continue
                    
                    s = json.dumps(new_item, ensure_ascii=False, indent=2)
                    f.write(s + ',\n')

        dummy = {'id': 'none', 'reviews':[]}
        s = json.dumps(dummy, ensure_ascii=False, indent=2)
        f.write(s + '\n')
        f.write(']')