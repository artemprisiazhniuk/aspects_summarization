import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import re
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def get_links(URL, links_limit=200):
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36'}

    links = []

    # page = requests.get(URL.format(0), headers=headers)
    # soup = BeautifulSoup(page.content, "html.parser")
    # res = soup.find('span', class_='qrwtg').text
    # num_items = int(''.join(re.findall(r"[0-9]+", res)))
    # if links_limit:
    #     num_items = min(num_items, links_limit)
    num_items = links_limit

    print(num_items)

    for i in tqdm(range(0, num_items, 30)):
        page = requests.get(URL.format(i), headers=headers)
        soup = BeautifulSoup(page.content, "html.parser")
        soup = soup.find('div', id='taplc_hsx_hotel_list_lite_dusty_hotels_combined_sponsored_undated_0')

        if not soup: continue

        for review in soup.findAll('a',{'class':'review_count'}):
            if review.has_attr('href'):
                a = review['href']
                a = 'https://www.tripadvisor.ru'+ a
                a = a[:(a.find('Reviews')+7)] + '-or{}' + a[(a.find('Reviews')+7):]
                links.append(a)

    links = list(set(links))
    assert len(links) > 0

    return links


def get_reviews(links, city, hotels_limit=200, reviews_limit=101):
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36'}
    class_to_rating = lambda x: int(x[-2:]) / 10
    tokenize = lambda x: re.findall(r"[\w']+|[.,!?;]+", x)

    with open(f'../data/data_hotels_{city}.jsonl', 'w+', encoding='utf-8') as f:
        f.write('[\n')

        if hotels_limit:
            links = links[:hotels_limit]
        for link in tqdm(links):
            new_item = {
                'id': link[link.find('or{}-')+5:link.find('.html')],
                'reviews': []
                }

            html2 = requests.get(link.format(0), headers=headers)
            item_soup = BeautifulSoup(html2.content,'lxml')
            if not item_soup: continue

            num_reviews = item_soup.find('span', class_='iypZC Mc _R b')
            if not num_reviews: continue
            num_reviews = int(num_reviews.text)
            if reviews_limit:
                num_reviews = min(num_reviews, reviews_limit)

            if num_reviews <= 0: continue
            
            for i in range(5, num_reviews, 5):
                html2 = requests.get(link.format(i), headers=headers)
                item_soup = BeautifulSoup(html2.content,'lxml')
                for r in item_soup.findAll('div', class_='YibKl MC R2 Gi z Z BB pBbQr'):
                    review_rating = class_to_rating(r.find('span', class_='ui_bubble_rating')['class'][-1])
                    review_text = r.find('q').text

                    if len(review_text) <= 0: continue

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