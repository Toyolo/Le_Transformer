import re
import random
import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List
import jax
import jax.numpy as jumpy
from flax import linen as nn
from jax import random
import optax


# Read the entire text file into a variable
with open('lettres_de_voyage.txt', 'r', encoding='utf-8') as file:
    full_text = file.read()

# Print the first 1000 characters
sample1K = full_text[:1000]
print(sample1K)
sample_length = len(sample1K)
print(f"Length of the 1K sample: {sample_length}")


def Le_tokenizer(text):
    text = full_text.lower()
    tokens = re.split(r"[ \t\n\r\f\v,.!?;:()\"“”]+", text)
    tokens = [token.strip() for token in tokens if token]

    return tokens

tokens = Le_tokenizer(full_text)
unitokes = list(set(tokens))
toke_to_index = {token: idx for idx, token in enumerate(unitokes)}
indicencies = [toke_to_index[token] for token in tokens]
unique_tokens = ['<PAD>'] + list(set(tokens))                                         #padded token set with lil nono zones
token_to_index = {token: idx for idx, token in enumerate(unique_tokens)}


pad_index = token_to_index['<PAD>']
indices = [token_to_index[token] for token in tokens]

sequence_length = 4
inputs = [indices[i:i+sequence_length] for i in range(len(indices) - sequence_length)]
targets = [indices[i+1:i+sequence_length+1] for i in range(len(indices) - sequence_length)]

max_sequence_length = max(len(seq) for seq in inputs)

inputs_padded = [seq + [pad_index] * (max_sequence_length - len(seq)) for seq in inputs]
targets_padded = [seq + [pad_index] * (max_sequence_length - len(seq)) for seq in targets]

input_masks = [[1 if idx != pad_index else 0 for idx in seq] for seq in inputs_padded]
target_masks = [[1 if idx != pad_index else 0 for idx in seq] for seq in targets_padded]

def create_batches(data, batch_size, shuffle=True):
  if shuffle:
    random.shuffle(data)

  batches = []
  for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    batches.append(batch)
  return batches

data = list(zip(inputs_padded, targets_padded, input_masks, target_masks))
batch_size = 36
batches = create_batches(data, batch_size)

@dataclass
class ModelConfig:
    block_size: int
    vocab_size: int
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4

class NewGELU(nn.Module):
  @nn.compact
  def __call__(self, x):
    return 0.5 * x * (1 + jax.nn.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * jumpy.power(x, 3))))


class ScaledDotProductAttention(nn.Module):
  d_k: int
  def __call__(self, q,k,v, mask=None):
    matmul_qk = jumpy.matmul(q,k.transpose(-2,-1))
    scaled_attention_logits = matmul_qk / jumpy.sqrt(self.d_k)

    if mask is not None:
      scaled_attention_logits += (mask * -1e9)
      attention_weights = nn.softmax(scaled_attention_logits, axis=1)
      output = jumpy.matmul(attention_weights, v)

      return output, attention_weights

class AttentionHydra(nn.Module):
  config: ModelConfig

  @nn.compact
  def __call__(self, q, k, v, mask=None):
    d_model = self.config.n_embd
    num_heads = self.config.n_head

    q, k, v = nn.Dense(features=d_model)(q), nn.Dense(features=d_model)(k), nn.Dense(features=d_model)(v)
    q, k, v = self.split_heads(q, num_heads), self.split_heads(k, num_heads), self.split_heads(v, num_heads)

    attention, attention_weights = ScaledDotProductAttention(d_k=d_model//num_heads)(q,k,v,mask)
    attention = self.concat_heads(attention)
    output = nn.Dense(features=d_model)(attention)

  def split_heads(self, x, num_heads):
    """if you cut off its head it will grow a head and then another head to be like woah i have 2 heads now get fucked huh"""
    x = x.reshape(x.shape[:1]+(num_heads, -1))
    return x.transpose(0,2,1,3)

  def concat_heads(self,x):
    """i could not think of a funny hydra metaphor for sticking all the heads back together maybe i need to read more"""
    x = x.transpose(0,2,1,3)
    return x.reshape(x.shape[:-2]+(-1,))

class Brick(nn.Module):
  config:ModelConfig

  @nn.compact
  def __call__(self, x, mask=None):
    x_ln1 = nn.LayerNorm()(x)
    attention_output = AttentionHydra(self.config)(x_ln1, x_ln1, x_ln1, mask)
    x = x + attention_output
    x_ln2 = nn.LayerNorm()(x)
    ffn_output = nn.Dense(features=self.config.n_embd*4)(x_ln2)
    ffn_output = NewGELU()(ffn_output)
    ffn_output = nn.Dense(features=self.config.n_embd)(ffn_output)

    x = x + ffn_output

    return x


class Le_Transformer(nn.Module):
  config: ModelConfig
  

  @nn.compact
  def __call__(self, input_ids, mask=None, deterministic=True):
    token_embedding = nn.Embed(self.config.vocab_size, self.config.n_embd)(input_ids)
    positional_embedding = nn.Embed(self.config.block_size,self.config.n_embd)(jumpy.arange(input_ids.shape[-1]))

    x = token_embedding + positional_embedding
    for _ in range(self.config.n_layer):
      x = Brick(self.config)(x,mask)
    x = nn.LayerNorm()(x)
    logits = nn.Dense(self.config.vocab_size)(x)

    return logits

def is_this_loss(logits, targets):
    one_hot_targets = jax.nn.one_hot(targets, num_classes=logits.shape[-1])
    loss = -jumpy.sum(one_hot_targets * jax.nn.log_softmax(logits), axis=-1)
    return jumpy.mean(loss)

learning_rate = 5e-4
optimizer = optax.adam(learning_rate)

@jax.jit
def train_step(optimizer, model, batch, rng):
    inputs, targets = batch

    def loss_fn(params):
        logits = model.apply({'params': params}, inputs, rngs={'dropout': rng})
        loss = is_this_loss(logits, targets)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grads)

    return optimizer, loss

@jax.jit
def test_step(model, batch, rng):
    inputs, targets = batch
    logits = model.apply({'params': model.params}, inputs, rngs={'dropout': rng})
    loss = is_this_loss(logits, targets)
    return loss


def generate_text(model, start_sequence, length, rng, temperature=1.0):
    generated = start_sequence
    for _ in range(length):
        logits = model.apply({'params': model.params}, generated, rngs={'dropout': rng})
        probs = jax.nn.softmax(logits / temperature)
        next_token = jax.random.categorical(rng, probs[-1])
        generated = jumpy.append(generated, next_token)

    return generated



