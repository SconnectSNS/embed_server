import torch
from transformers import DistilBertTokenizer, DistilBertModel
from flask import Flask, jsonify, request
from flask_cors import CORS
import os

# 사전 학습된 DistilBERT 모델과 토크나이저를 로드합니다.
model_dir = "/var/task/model"
tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
model = DistilBertModel.from_pretrained(model_dir)

app = Flask(__name__)
CORS(app)

@app.route('/embed', methods=['POST'])
def embed():
    # POST 방식으로 전송된 데이터를 추출합니다.
    data = request.json
    keywords = data['keywords']

    # 기존 DB에 존재하던 게시글들과 새로 들어온 게시글의 키워드 중 가장 긴 값을 고릅니다.
    max_length = max(tokenizer(keyword, truncation=True, return_tensors="pt")['input_ids'].shape[-1] for keyword in keywords)

    inputs = []
    # 기존 DB에 존재하던 게시글들의 토큰을 구합니다.
    for keyword in keywords:
        inputs.append(tokenizer(keyword, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt"))

    # 전체 토큰의 embedding을 구합니다.
    outputs = [model(**input_) for input_ in inputs]
    keyword_embeddings = torch.stack([output.last_hidden_state.mean(dim=1) for output in outputs])

    # 키워드 집합 간의 cosine similarity를 계산합니다.
    similarity_matrix = torch.zeros(len(keywords), len(keywords))
    similarity_matrix = [[[j,0] for j in range(len(keywords))] for _ in range(len(keywords))]

    for i in range(len(keywords)):
        for j in range(len(keywords)):
            if i == j:
                continue
            else:
                similarity_matrix[i][j][1] = torch.cosine_similarity(keyword_embeddings[i], keyword_embeddings[j])

    for i in range(len(keywords)):
        # cosine similarity가 큰 순서대로 정렬합니다.
        similarity_matrix[i] = sorted(similarity_matrix[i], key=lambda x: x[1],reverse=True)

    results = []

    for i in range(len(keywords)):
        results += [{'index': [k[0] for k in similarity_matrix[i][:5]]}]

    return jsonify(results)

if __name__ == '__main__':
    print('Server Run')
    app.run(host='0.0.0.0', port='9090', debug=True)
