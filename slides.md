<!-- Treat as reveal.js with no part slides (heading 1 is a new slide) -->

<style>
:root {
  --hadron-bg-1: #060a14;
  --hadron-bg-2: #0d1730;
  --hadron-text: #edf2ff;
  --hadron-muted: #b7c5e6;
  --hadron-cyan: #10c6ea;
  --hadron-magenta: #ff2d9a;
  --hadron-yellow: #f4e54b;
}

.reveal {
  color: var(--hadron-text);
  font-family: "Inter", "Aptos", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 34px;
}

.reveal .slides {
  text-align: left;
}

.reveal .slides section {
  padding: 0.4em 0.8em 0.6em 0.8em;
  background:
    radial-gradient(circle at 12% 14%, rgba(16, 198, 234, 0.12), transparent 23%),
    radial-gradient(circle at 88% 18%, rgba(255, 45, 154, 0.11), transparent 24%),
    radial-gradient(circle at 82% 84%, rgba(244, 229, 75, 0.08), transparent 18%),
    linear-gradient(145deg, var(--hadron-bg-1) 0%, var(--hadron-bg-2) 100%);
}

.reveal h1, .reveal h2, .reveal h3 {
  color: #ffffff;
  letter-spacing: -0.03em;
  margin-bottom: 0.4em;
}

.reveal h1 { font-size: 1.55em; }
.reveal h2 { font-size: 1.2em; }

.reveal p, .reveal li {
  color: var(--hadron-text);
  line-height: 1.28;
}

.reveal strong { color: #ffffff; }

.reveal ul, .reveal ol {
  display: block;
  margin-left: 0.8em;
}

.reveal pre {
  width: 100%;
  margin: 0.35em 0 0.55em 0;
  box-shadow: none;
}

.reveal pre code {
  max-height: 420px;
  padding: 0.75em 0.9em;
  font-size: 0.74em;
  line-height: 1.3;
  color: #eff5ff;
  background: rgba(10, 16, 32, 0.9);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 14px;
}

.reveal code {
  color: #cfe3ff;
}

.reveal img {
  border: none;
  box-shadow: none;
  background: transparent;
}

.reveal blockquote {
  width: 100%;
  margin: 0.5em 0;
  padding: 0.5em 0.8em;
  color: var(--hadron-text);
  background: rgba(255,255,255,0.06);
  border-left: 4px solid var(--hadron-cyan);
  box-shadow: none;
}

.title-slide {
  margin-top: 0.3em;
}

.title-slide .org {
  margin-bottom: 0.6em;
  font-size: 0.56em;
  font-weight: 700;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: #a9c4ff;
}

.title-slide .title-grid {
  display: grid;
  grid-template-columns: 1.2fr 0.95fr;
  gap: 1.1em;
  align-items: center;
}

.title-slide .meta {
  margin-top: 1.2em;
  font-size: 0.63em;
  line-height: 1.45;
  color: var(--hadron-muted);
}

.title-slide .logo-card {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 170px;
  padding: 0.7em 0.9em;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 22px;
}

.title-slide .hadron-logo {
  width: 100%;
  max-width: 470px;
}

.reveal hr {
  border: none;
  border-top: 1px solid rgba(255,255,255,0.12);
}
</style>

# Interesting AI

<div class="title-slide">
  <div class="org">Hadron Industries</div>
  <div class="title-grid">
    <div class="title-copy">
      <div class="meta">
        <strong>Sam Hartman</strong><br>
        October 30, 2026
      </div>
    </div>
    <div class="logo-card">
      <img class="hadron-logo" alt="Hadron Industries logo" src="hadron-logo.png">
    </div>
  </div>
</div>

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

The tokenizer makes PyTorch tensors, then we move them to the model's GPU. Token boundaries are not necessarily word boundaries; inspect them with `t.convert_ids_to_tokens(input['input_ids'][0].tolist())`.

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
