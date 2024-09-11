from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/api/search', methods=['GET'])
def search():
    term = request.args.get('term')
    # 여기에 검색어 처리 로직을 추가하세요
    results = {"message": f"Search term received: {term}"}
    return jsonify(results)

# 특정 API에서 데이터를 가져오거나, 서버의 상태 정보를 반환하거나, 데이터베이스에서 특정 값을 가져오는 등의 역할
@app.route('/api/endpoint', methods=['GET'])
def fetch_data():
    # 데이터를 처리하거나 가져오는 로직 추가
    data = {"message": "Data fetched successfully!"}  # 데이터를 JSON으로 반환
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5000)

    ##1234

