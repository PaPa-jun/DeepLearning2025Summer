import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import numpy as np

# 1. 加载数据
data_frame = pd.read_json("train.jsonl", lines=True)
texts = data_frame["text"].tolist()
labels = data_frame["label"].tolist()

# 2. 使用 BERT 提取文本特征
# 加载预训练 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to("cuda" if torch.cuda.is_available() else "cpu")
bert_model.eval()  # 设置为评估模式

# 定义 BERT 特征提取函数
def get_bert_features(texts):
    features = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(bert_model.device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        # 提取 [CLS] 标记的嵌入向量作为文本特征
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        features.append(cls_embedding[0])
    return np.array(features) 

# 生成 BERT 特征
X = get_bert_features(texts)
print("BERT 特征维度:", X.shape)  # 输出特征维度

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 4. 定义 XGBoost 模型和参数网格
xgb_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
param_grid = {
    'max_depth': [3, 5, 7],           # 树的最大深度
    'learning_rate': [0.01, 0.1, 0.2], # 学习率
    'n_estimators': [50, 100, 150]     # 树的数量
}

# 5. 网格搜索调参
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='f1',  # 使用 F1 分数优化
    cv=3,          # 3 折交叉验证
    verbose=1,
    n_jobs=-1      # 并行计算
)
grid_search.fit(X_train, y_train)

# 6. 最佳模型评估
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("最佳参数:", grid_search.best_params_)  # 输出最佳参数
print("分类报告:\n", classification_report(y_test, y_pred))

# 7. 测试集预测并保存结果
test_data = pd.read_json("test.jsonl", lines=True)
test_texts = test_data["text"].tolist()
X_test_features = get_bert_features(test_texts)
y_pred = best_model.predict(X_test_features)

with open("submit.txt", "w") as file:
    file.write("\n".join(map(str, y_pred)))