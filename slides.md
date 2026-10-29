<!-- Treat as reveal.js with no part slides (heading 1 is a new slide) -->

# Interesting AI

Sam Hartman

October 30, 2026

# Loading a model
```python
import transformers, torch

model_id = 'google/gemma-4-E4B-it'
m = transformers.AutoModelForCausalLM.from_pretrained(model_id).to('cuda')
t = transformers.AutoTokenizer.from_pretrained(model_id)
```

`m` is the pretrained model, placed on the GPU; `t` is its matching tokenizer.

# Tokenizing a prompt

Models consume **token IDs** rather than raw text. A token may be a word, part of a word, punctuation, whitespace, or a special marker.

```python
input = t('Python is a ', return_tensors='pt')
input = {k:v.to('cuda') for k,v in input.items()}
```

The tokenizer makes PyTorch tensors; the second line moves them to the same GPU as the model.

# Calling the model: output shape

```python
output = m(**input)
output.logits.shape
# torch.Size([1, 4, 262144])
```

- **1**: one input sequence (batch size).
- **4**: four token positions in that sequence.
- **262,144**: one score for each possible token ID at every position.

This is one forward pass, **not** a generated sentence.

# What are logits?

A **logit** is an unnormalized score for a candidate next token.

- Higher score means the model favors that candidate more.
- Logits are **not probabilities** (softmax can turn them into probabilities).
- The final position tells us what token might follow our prompt.

```python
output.logits[0, -1]  # scores at the last input position
```

`0` chooses our only input sequence; `-1` chooses its last token position.

# Finding the top ten predictions

```python
topten = torch.topk(output.logits[0,-1], 10)
[t.decode(x) for x in topten.indices]
```

`torch.topk` returns the ten largest logit **values** and their token **indices** (IDs). The tokenizer decodes each ID back into readable text.

This is a **ranking**, not random sampling or text generation.

# The model's top ten next-token candidates

Prompt: `Python is a `

```python
['1', '\n', '2', '4', '5', '0', '🐍', ' Python', '3', '\n\n']
```

The candidates are in descending score order. Spaces and newlines matter: `' Python'` starts with a space, while `'\n'` and `'\n\n'` represent line breaks.

These are **alternatives for a single next token**, not a ten-token continuation.
