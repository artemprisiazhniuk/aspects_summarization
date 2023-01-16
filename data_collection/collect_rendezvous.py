import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import re
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def get_links(URL, links_limit=None):
    links = []
    ids = []

    # URL = "https://www.rendez-vous.ru/catalog/odezhda/page/{}"
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36'}

    page = requests.get(URL.format(1), headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")

    pages = int(soup.findAll('li', class_='page')[-1].find('a').text)
    if links_limit:
        pages = min(pages, links_limit)

    for i in tqdm(range(1, pages)):
        page = requests.get(URL.format(i), headers=headers)
        soup = BeautifulSoup(page.content, "html.parser")

        for item in soup.findAll('li', class_='item'):
            id = item['data-code']
            ids.append(id)
            a = item.find('a', class_='item-link')['href']
            a = 'https://www.rendez-vous.ru'+ a
            links.append(a)

    links = list(set(links))
    ids = list(set(ids))

    return links, ids


def get_reviews(links, ids, items_limit=None, reviews_limit=None):
    # URL = "https://www.rendez-vous.ru/catalog/odezhda/futbolka/calzetti_top2f_noodles1_chernyy-3348035/"

    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36'}
    tokenize = lambda x: re.findall(r"[\w']+|[.,!?;]+", x)
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)

    with open('../data/data_clothes_rendezvous.jsonl', 'w+', encoding='utf-8') as f:
        if items_limit: links = links[:items_limit]
        for j in tqdm(range(len(links))):
            url = links[j]
            id = ids[j]

            session = requests.Session()

            page = requests.get(url, headers=headers)
            soup = BeautifulSoup(page.content, "html.parser")

            headers = {
                'user-agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                "x-requested-with": "XMLHttpRequest"
            }
            data = {
                'limit': reviews_limit,
                'offset': 0,
                'model_id': id
            }
            res = session.post(
                'https://www.rendez-vous.ru/ajax/getReviews/', 
                headers=headers, 
                data=data,
                verify=False
                )

            if res.text:
                res = json.loads(res.text)

                new_item = {'id': id, 'reviews': []}

                soup = BeautifulSoup(res['data'], "html.parser")    
                for review in soup.findAll('div', class_='reviews__item'):
                    review_rating = review.find('div', class_='rating-bg')
                    review_rating = int(re.findall(r'[0-9]+', review_rating['style'])[0]) / 20 # 100 -> 5.0

                    review_text = review.find('p', {'itemprop': 'reviewBody'}).text
                    review_text = emoji_pattern.sub(r'', review_text)
                    review_sentences = [' '.join(tokenize(t)) for t in sent_tokenize(review_text, language='russian')]

                    if not review_sentences: continue # skip if empty review

                    new_review = {'sentences': review_sentences, 'rating': review_rating}
                    new_item['reviews'].append(new_review)

                if len(new_item['reviews']) <= 0: continue
                json.dump(new_item, f, ensure_ascii=False, indent=2)