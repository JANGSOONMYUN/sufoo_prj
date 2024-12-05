import requests
import shutil
import json
import os

# JSON 파일에서 API_KEY 불러오기
with open('modules/search_img/config.json', 'r') as config_file:
    config = json.load(config_file)
    API_KEY = config['api_key']

BASE_URL = 'https://pixabay.com/api/'

def search_images(query, page=1, per_page=200):
    params = {
        'key': API_KEY,
        'q': query,
        'image_type': 'all',
        'orientation': 'all',
        'category': 'background',
        'min_width': 0,
        'min_height': 0,
        'colors': '',
        'editors_choice': 'false',
        'safesearch': 'false',
        'order': 'popular',
        'page': page,
        'per_page': per_page,
        'lang': 'en'
    }
    response = requests.get(BASE_URL, params=params)
    return response.json()

def download_image(url, file_name):
    response = requests.get(url, stream=True)
    with open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
            
def search_and_download(keyword, num_imgs, download_dir):
    image_name_list = []
    j = 1
    for n in range(1, 2):
        ims = search_images(keyword, page=n)
        for i in range(len(ims['hits'])):
            payload = ims['hits'][i]['largeImageURL']
            filename = f"{keyword}_{j}.jpg"
            download_image(payload, os.path.join(download_dir, filename))
            image_name_list.append(filename)
            print(f"{j} URL of image: {payload}")
            j += 1
            
            if j > num_imgs:
                return image_name_list
    return image_name_list
    
def test():
    j = 1
    for n in range(1, 2):
        ims = search_images("시서스 가루", page=n)
        for i in range(len(ims['hits'])):
            payload = ims['hits'][i]['largeImageURL']
            download_image(payload, f"{j}_local_image.jpg")
            print(f"{j} URL of image: {payload}")
            j += 1

if __name__ == "__main__":
    search_and_download('orange', 2, '/home/jsm/llm/sufoo_prj/llm_agent/images')
    # test()