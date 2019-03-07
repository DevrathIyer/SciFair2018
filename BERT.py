import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenized input
text = "The earbuds were great, but lacked bass"
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Predict hidden states features for each layer
encoded_layers, _ = model(tokens_tensor)

# Load pre-trained model (weights)

# Predict all tokens
predictions = model(tokens_tensor)

# confirm we were able to predict 'henson'
print(predictions)