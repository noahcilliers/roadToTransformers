import urllib.request
import re
from importlib.metadata import version
import tiktoken

import torch
from torch.utils.data import Dataset, DataLoader


## this efficiently loads our txt into chunks for us to train over

class GPTDatasetV1(Dataset):
	def __init__(self, txt, tokenizer, max_length, stride):
		self.input_ids = []
		self.target_ids = []

		token_ids = tokenizer.encode(txt)

		for i in range(1, len(token_ids) - max_length, stride):
			input_chunk = token_ids[i:i + max_length]
			target_chunk = token_ids[i+1: i + 1 + max_length]

			self.input_ids.append(torch.tensor(input_chunk))
			self.target_ids.append(torch.tensor(target_chunk))

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, idx):
		return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
		tokenizer = tiktoken.get_encoding("gpt2")
		dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

		return dataloader


class Tokenizer():
	def __init__(self, vocab):
		self.vocab = vocab
		self.int_to_str = {i:s for s,i in vocab.items()}

	def encode(self, text):
		# process text into tokens
		toks = re.split(r'([,.:;?_!"()\']|--|\s)', text)
		toks = [item for item in toks if item.strip()]
		# tokens into ids
		ids = []
		for tok in toks:
			if tok in self.vocab:
				ids.append(self.vocab[tok])
			else:
				ids.append(self.vocab["<|unk|>"])

		return ids
		

	def decode(self, ids):
		text = " ".join([self.int_to_str[i] for i in ids])
		text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
		return text


def init_vocab(text):
	preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
	preprocessed = [item for item in preprocessed if item.strip()]
	all_tokens = sorted(set(preprocessed))
	all_tokens.extend(["<|endoftext|>", "<|unk|>"])
	vocab = {token:integer for integer,token in enumerate(all_tokens)}
	vocab_size = len(all_tokens)
	return vocab, vocab_size


def softmax_naive(x):
	return torch.exp(x) / torch.exp(x).sum(dim=0)


def main():
	"""
	print("tiktoken version:", version("tiktoken"))
	with open("the-verdict.txt", "r", encoding="utf-8") as f:
		raw_text = f.read()
	

	dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
	data_iter = iter(dataloader)
	
	vocab_size = 50257
	output_dim = 256

	token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

	inputs, _ = next(data_iter)
	embs = token_embedding_layer(inputs)
	print(embs.shape)
	
	context_length = 4
	pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
	pos_embs = pos_embedding_layer(torch.arange(context_length))
	print(pos_embs.shape)

	input_embs = embs + pos_embs
	print(input_embs.shape)
	"""

	#################################
	# ATTENTION STARTS HERE
	#################################
	inputs = torch.tensor(
		[[0.43, 0.15, 0.89],
		[0.55, 0.87, 0.66],
		[0.57, 0.85, 0.64],
		[0.22, 0.58, 0.33],
		[0.77, 0.25, 0.10],
		[0.05, 0.80, 0.55]]
	)

	# calculate context vectors for all inputs
	# same shape as the inputs vector
	# init and get attention scores
	context_length = 6
	emb_dim = 3
	
	attn_scores = torch.zeros(context_length, context_length)
	for i in range(context_length):
		for j in range(context_length):
			attn_scores[i][j] = torch.dot(inputs[i], inputs[j])

	# now we have attn scores of shape context x context
	# we shall normalize to make them add to 1
	# dim = -1 makes them normalize to the rows
	attn_weights = torch.softmax(attn_scores, dim=-1)
	


	context = attn_weights @ inputs
	









	"""
	# We want to calculate the attention score for our second input vec
	# We have to calculate the dot product between our query token and all other tokens
	query = inputs[1]
	attn_score_2 = torch.empty(inputs.shape[0])
	for i, x_i in enumerate(inputs):
		attn_score_2[i] = torch.dot(query, x_i)
	
	print(attn_score_2)

	#Attention scores are not normalized, but attention weights are normalized
	# The dot product is the key here because find one attnetion score in between each q, k

	### Normalize all the weights to equal one
	attn_weights_2 = torch.softmax(attn_score_2, dim=0)

	context_vec_2 = torch.empty(query.shape)
	for i, x_i in enumerate(inputs):
		context_vec_2 += attn_weights_2[i] * x_i

	print(context_vec_2)
	"""


if __name__ == "__main__":
	main()