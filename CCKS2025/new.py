import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import nlpaug.augmenter.word as naw  # 文本增强 [[8]]

# 1. 加载数据
data_frame = pd.read_json("train.jsonl", lines=True)
texts = data_frame["text"].tolist()
labels = data_frame["label"].tolist()

# 2. 数据增强（同义词替换） [[7]]
aug = naw.SynonymAug(aug_src='wordnet')
augmented_texts = [aug.augment(text)[0] for text in texts]  # 增强数据
texts += augmented_texts
labels += labels  # 假设标签不变

# 3. 使用BERT提取文本特征（平均池化）
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to("cuda" if torch.cuda.is_available() else "cpu")
bert_model.eval()

def get_bert_features(texts):
    features = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(bert_model.device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        # 平均池化所有token的嵌入向量 [[1]]
        mean_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        features.append(mean_embedding[0])
    return np.array(features)

X = get_bert_features(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

# 4. 参数调优与正则化
xgb_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
param_grid = {
    'max_depth': [5],              # 通过网格搜索确定最优深度
    'learning_rate': [0.1],        # 学习率
    'n_estimators': [100],         # 树的数量
    'reg_alpha': [0.1, 0.5],       # L1正则化 [[1]]
    'reg_lambda': [0.1, 0.5]       # L2正则化
}

cv = StratifiedKFold(n_splits=5)  # 分层交叉验证
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='f1',
    cv=cv,
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# 5. 集成学习（Stacking）
stacking_model = StackingClassifier(
    estimators=[
        ('xgb', grid_search.best_estimator_),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))  # 添加随机森林基模型
    ],
    final_estimator=LogisticRegression()  # 元模型
)
stacking_model.fit(X_train, y_train)

# 6. 最终评估
y_pred = stacking_model.predict(X_test)
print("最佳参数:", grid_search.best_params_)
print("分类报告:\n", classification_report(y_test, y_pred))

# 7. 测试集预测
test_data = pd.read_json("test.jsonl", lines=True)
test_texts = test_data["text"].tolist()
X_test_features = get_bert_features(test_texts)
y_pred = stacking_model.predict(X_test_features)

with open("submit.txt", "w") as file:
    file.write("\n".join(map(str, y_pred)))