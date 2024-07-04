"""
Eliminating copies of definitions
"""
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pandas as pd

data = pd.read_csv("vector_db.csv")
data = data.drop(data.keys()[0], axis=1)
res = []
tensor = torch.tensor
for i in range(len(data)):
    data.loc[i, "Embeding"] = list(eval(data["Embeding"][i]))

res = []
max_ = 0.99
for i in range(len(data['Embeding'])):
    for j in range(i + 1, len(data['Embeding'])):
        x = cosine_similarity(data['Embeding'][i], data["Embeding"][j])[0][0]
        if x > max_:
            print(data["name"][i])
            print(data["name"][j])
            res.append(i)
            print(i, j, x)
data = data.drop(res)

data.to_csv("db_cleaned.csv")
