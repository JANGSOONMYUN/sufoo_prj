import sys
import os

# 현재 파일의 경로
file_path = os.path.abspath(__file__)
# 현재 파일이 있는 디렉토리의 경로
directory_path = os.path.dirname(file_path)
# sys.path에 해당 경로 추가
if directory_path not in sys.path:
    sys.path.append(directory_path)
    
from search_img_api import search_and_download

def include_images(data):
    request_list = []
    if 'request' in data:
        request_list = data['request']
        
    download_dir = '/home/jsm/llm/sufoo_prj/llm_agent/images'
    open_link_url = 'http://jsm0803.iptime.org:20000/images/'
    
    for i, req in enumerate(request_list):
        if 'representative_image_name' not in req:
            continue
        image_name = req['representative_image_name']
        image_name_list = search_and_download(keyword=image_name, num_imgs=1, download_dir=download_dir)
        image_url = open_link_url + image_name_list[0]
        request_list[i]['image_url'] = image_url
    
    print('#'*100)
    print(request_list)
    print(data)
    data['request'] = request_list
    return data
    
    