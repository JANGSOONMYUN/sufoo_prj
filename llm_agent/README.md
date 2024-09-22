# Requirements
- python >= 3.9
- pytorch for CPU only
### pip install
- openai
- transformers
- websockets
- flask
- gunicorn
#### LangChain
- pip install langchain-community==0.2.9 langchain==0.2.9 langchain-core==0.2.22 langchain-openai==0.1.10
### Example
```
conda create -n openai python=3.9
conda activate openai
pip install openai transformers websockets pyinstaller flask flask_cors pip install Flask[async] gunicorn
conda install pytorch cpuonly -c pytorch
```

### Extra modules for server
sudo apt update
sudo apt install git

- anaconda (https://www.anaconda.com/products/individual)
```
sudo wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash [Anaconda*.sh]
source ~/.bashrc
```

# Configuration
- File name = [settings/config.json]
- File contents = 
```
{
    "api_key": "sk-JEP5T......",
    "api_key_0": "sk-JEP5T......",
    ...,
    "port": 12009,
    "close_server_when_client_die": true 
}
```
- "api_key": API key of OpenAI; 현재 하나의 key 만 공통적으로 사용.
- "port": port number
- "close_server_when_client_die": 클라이언트 연결이 끊겼을 때 서버의 상태. (true: 서버 종료, false: 서버 유지) [deprecated]


# How to run
### Flask (Single connection for test) [deprecated]
```
export FLASK_APP=server_flask.py
flask run --host=0.0.0.0 --port=10000 

# windows
$env:FLASK_APP = "server_flask.py"
flask run --host=0.0.0.0 --port=10000 
```
### Gunicorn [deprecated]
```
gunicorn -w 1 --bind 0.0.0.0:10000 server_flask:app &
gunicorn -w 4 --bind 0.0.0.0:10000 server_flask:app &
```
- Close gunicorn process
```
ps -ef | grep gunicorn
pgrep -f "gunicorn"
kill [pid]
```
- Close all
```
pkill -f "gunicorn"
```
#### Kill processes
```
# example port number = 15000
# Install lsof
sudo apt install lsof
# Check processes occupying a port number
lsof -i :15000
# Kill corresponding processes
lsof -ti :15020 | xargs kill
# OR
lsof -i :15019 | awk 'NR!=1 {print $2}' | xargs kill -9
```