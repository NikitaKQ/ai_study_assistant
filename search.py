"""
Vectorization and search
"""
import random
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Set a random seed
random_seed = 42
random.seed(random_seed)

# Set a random seed for PyTorch (for GPU as well)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def vectorize(text):
    # Tokenize and encode text using batch_encode_plus
    # The function returns a dictionary containing the token IDs and attention masks
    encoding = tokenizer.batch_encode_plus(
        [text],  # List of input texts
        padding=True,  # Pad to the maximum sequence length
        truncation=True,  # Truncate to the maximum sequence length if necessary
        return_tensors='pt',  # Return PyTorch tensors
        add_special_tokens=True  # Add special tokens CLS and SEP
    )

    with torch.no_grad():
        outputs = model(encoding['input_ids'], attention_mask=encoding['attention_mask'])
        word_embeddings = outputs.last_hidden_state  # This contains the embeddings

    # Compute the average of word embeddings to get the sentence embedding
    sentence_embedding = word_embeddings.mean(dim=1)  # Average pooling along the sequence length dimension

    return sentence_embedding


if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("db.csv")
    data = data.drop(data.keys()[0], axis=1)
    res = []
    for name, _ in data.values:
        res.append(vectorize(name))
    data["Embeding"] = res
    data.to_csv("vector_db.csv")