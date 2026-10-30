---
pagetitle: Interesting AI
revealjs-url: 'https://unpkg.com/reveal.js@4.1.0'
css: hadron.css
---

<!-- Treat as reveal.js with no part slides (heading 1 is a new slide) -->

# Interesting AI

::: {.title-slide}
::: {.org}
Hadron Industries
:::

::: {.title-grid}
::: {.title-copy}
**Sam Hartman**

October 30, 2026
:::

::: {.logo-card}
<img src="hadron-logo.png" alt="Hadron Industries logo" class="hadron-logo">
:::
:::
:::

# How LLMs Predict the Next Token

![Two-panel diagram explaining next-token prediction for “2 + 2 = 4”. Panel 1 is titled “Step 1: Before ‘=’.” It shows input tokens 2, plus, 2 flowing into a box labeled LLM. To the right is a vertical list titled “Next-token probabilities,” where the token “=” has the highest probability (about 0.65) and several other tokens such as plus, period, digit 2, and <newline> have smaller probabilities. Panel 2 is titled “Step 2: After ‘=’.” It shows tokens 2, plus, 2, equals flowing into the LLM box. The output side shows next-token probabilities where “4” has very high probability (around 0.92) and alternative tokens like 3, 22, and 5 have much lower probabilities. At the bottom, an example output sequence 2 + 4 is shown, with a caption explaining that once the “=” token is included the LLM predicts the result with high probability.](llm-next-token-prediction.png) 

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

The tokenizer makes PyTorch tensors, then we move them to the model's GPU. Token boundaries are not necessarily word boundaries; inspect them with `t.convert_ids_to_tokens`.

# Calling the model: why four positions?

```python
output = m(**input)
output.logits.shape
# torch.Size([1, 4, 262144])
```

The tokenizer produced **four input token IDs** from `'Python is a '` (tokens are not necessarily words).

- **1**: one input sequence.
- **4**: one position per input token. Each position sees its prefix and scores the *following* token.
- **262,144**: one score for every possible next token at each position.

**One forward pass computes all four positions.** The last one, `output.logits[0, -1]`, predicts what comes **after the entire prompt**.

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

# SFT: Before training

<img src="sft_before.png" class="training-landscape-image" alt="Before training: A cyan training-example path crosses a three-dimensional blue probability landscape. The model's magenta preferred continuation initially follows the example, then diverges into a different valley when it favors another next token.">

# SFT: After training

<img src="sft_after_training.png" class="training-landscape-image" alt="After training: The cyan example follows the same route, but the landscape has changed. A new valley favors the desired continuation, while the old magenta preference is shown faintly on raised terrain. The example stays fixed while training adjusts the model.">
