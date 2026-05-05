---
name: nlp-text-analysis
description: Text data analysis for NLP research — dataset characterization, preprocessing decision guide, tokenization choices, embedding selection, and evaluation metrics (BLEU, ROUGE, BERTScore, perplexity). Use when analyzing or processing text datasets.
allowed_agents: [data, experiment]
---

# NLP Text Analysis

## Overview

This skill covers the full workflow for analyzing and preprocessing text datasets in NLP research: profiling raw data, making principled preprocessing decisions, selecting tokenizers and embeddings, and evaluating model outputs with standard metrics. Apply this skill whenever working with text corpora, NLP benchmarks, or language model outputs.

## When to Use This Skill

Use this skill when:
- Exploring a new text dataset before modeling (corpus statistics, quality checks)
- Deciding how to preprocess text (tokenization, normalization, cleaning)
- Choosing an embedding strategy for a downstream task
- Evaluating NLP model outputs (translation, summarization, QA, classification)
- Selecting benchmark datasets for a given NLP task
- Augmenting a small text dataset to improve generalization

---

## Text Dataset Characterization

Before any modeling, profile your corpus systematically. Use `scripts/text_profiler.py` for automated profiling, or run the analyses below interactively.

### Core Statistics

```python
import pandas as pd
import collections
import re

# Load dataset (CSV, TSV, or JSONL)
df = pd.read_csv("dataset.csv")  # or pd.read_json("data.jsonl", lines=True)
texts = df["text"].dropna().tolist()

# Basic whitespace tokenization for profiling
def simple_tokenize(text):
    return text.lower().split()

tokens_per_doc = [simple_tokenize(t) for t in texts]
token_lengths = [len(t) for t in tokens_per_doc]
char_lengths = [len(t) for t in texts]

# Token count distribution
import numpy as np
print(f"Total documents:    {len(texts)}")
print(f"Total tokens:       {sum(token_lengths):,}")
print(f"Avg tokens/doc:     {np.mean(token_lengths):.1f}")
print(f"Median tokens/doc:  {np.median(token_lengths):.1f}")
print(f"Max tokens/doc:     {max(token_lengths)}")
print(f"Min tokens/doc:     {min(token_lengths)}")
print(f"Std tokens/doc:     {np.std(token_lengths):.1f}")

# Vocabulary size
all_tokens = [tok for doc in tokens_per_doc for tok in doc]
vocab = collections.Counter(all_tokens)
print(f"\nVocabulary size (whitespace, lowercased): {len(vocab):,}")
print(f"Singleton tokens (freq=1):               {sum(1 for v in vocab.values() if v == 1):,}")

# Top 20 most frequent tokens
print("\nTop 20 tokens:")
for tok, freq in vocab.most_common(20):
    print(f"  {tok:<20} {freq:>8,}")
```

### OOV Rate Against a Reference Vocabulary

```python
# Load a reference vocab (e.g., from a pre-trained model's vocabulary file)
with open("vocab.txt") as f:
    ref_vocab = set(line.strip() for line in f)

oov_count = sum(1 for tok in all_tokens if tok not in ref_vocab)
oov_rate = oov_count / len(all_tokens)
print(f"OOV rate vs reference vocab: {oov_rate:.2%}")
```

### Sequence Length Distribution

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(token_lengths, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(512, color='red', linestyle='--', label='512 token limit')
plt.xlabel("Token length")
plt.ylabel("Count")
plt.title("Token Length Distribution")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(char_lengths, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel("Character length")
plt.ylabel("Count")
plt.title("Character Length Distribution")
plt.tight_layout()
plt.savefig("length_distribution.png", dpi=150)
plt.show()

# Percentage of docs exceeding transformer limits
pct_over_512 = sum(1 for l in token_lengths if l > 512) / len(token_lengths)
pct_under_5 = sum(1 for l in token_lengths if l < 5) / len(token_lengths)
print(f"Docs > 512 tokens:  {pct_over_512:.2%}")
print(f"Docs < 5 tokens:    {pct_under_5:.2%}")
```

### Label / Class Distribution

```python
if "label" in df.columns:
    label_dist = df["label"].value_counts()
    print("\nLabel distribution:")
    print(label_dist.to_string())

    # Check imbalance
    majority = label_dist.max()
    minority = label_dist.min()
    imbalance_ratio = majority / minority
    print(f"\nImbalance ratio (majority/minority): {imbalance_ratio:.1f}x")
    if imbalance_ratio > 5:
        print("WARNING: Severe class imbalance detected. Consider oversampling, "
              "undersampling, or class-weighted loss.")
```

### Language Detection

```python
# pip install langdetect
from langdetect import detect, LangDetectException
import collections

lang_counts = collections.Counter()
for text in texts[:2000]:  # sample for speed
    try:
        lang_counts[detect(text)] += 1
    except LangDetectException:
        lang_counts["unknown"] += 1

print("Detected languages (sample of 2000):")
for lang, count in lang_counts.most_common():
    print(f"  {lang}: {count}")
```

### Encoding / Quality Issues

```python
import unicodedata

issues = {
    "duplicates": df.duplicated(subset=["text"]).sum(),
    "null_texts": df["text"].isna().sum(),
    "empty_strings": (df["text"].str.strip() == "").sum(),
}

# Detect potential encoding artifacts
mojibake_pattern = re.compile(r'[â€™â€œâ€]')
issues["potential_mojibake"] = df["text"].str.contains(
    mojibake_pattern, regex=True, na=False
).sum()

# Texts with high non-ASCII ratio
def non_ascii_ratio(text):
    if not text:
        return 0
    return sum(1 for c in text if ord(c) > 127) / len(text)

df["non_ascii_ratio"] = df["text"].apply(lambda t: non_ascii_ratio(str(t)))
issues["high_non_ascii"] = (df["non_ascii_ratio"] > 0.3).sum()

for key, val in issues.items():
    print(f"{key}: {val}")
```

---

## Preprocessing Decision Guide

### Lowercasing

| Scenario | Decision |
|---|---|
| Bag-of-words, TF-IDF, topic modeling | YES — reduces sparsity |
| Named Entity Recognition | NO — case is a feature |
| Fine-tuned BERT (cased model) | NO — use cased tokenizer |
| Fine-tuned BERT (uncased model) | YES — model expects lowercase |
| Sentiment analysis | CAREFUL — "GREAT" vs "great" carry different signals |

```python
text = text.lower()  # only when appropriate
```

### Punctuation Removal

| Scenario | Decision |
|---|---|
| Topic modeling, keyword extraction | YES |
| Sentiment analysis | NO — "!" and "?" carry sentiment |
| Syntactic parsing | NO — punctuation is grammatically meaningful |
| Machine translation | NO — punctuation is part of the output |

```python
import re
text_no_punct = re.sub(r'[^\w\s]', '', text)  # removes all non-word, non-space
```

### Stopword Removal

| Scenario | Decision |
|---|---|
| TF-IDF retrieval, topic modeling | YES — reduces noise |
| Language model training | NO — stopwords are part of grammar |
| BERT fine-tuning | NO — BERT expects natural text |
| Short text classification | CAREFUL — removing words from short texts can hurt |

```python
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
tokens = [tok for tok in tokens if tok not in stop_words]
```

### Stemming vs Lemmatization

| Method | Speed | Quality | Use Case |
|---|---|---|---|
| Stemming (Porter/Snowball) | Fast | Crude (dogs → dog, running → run) | High-volume retrieval, when speed > quality |
| Lemmatization (WordNet, spaCy) | Slower | Accurate (better → good) | Semantic analysis, when quality matters |

```python
# Stemming (fast)
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed = [stemmer.stem(tok) for tok in tokens]

# Lemmatization (accurate)
import nltk
nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(tok) for tok in tokens]
```

### HTML, URL, and Emoji Stripping

```python
import re
import html

def clean_text(text: str, strip_html=True, strip_urls=True, strip_emojis=False) -> str:
    """Flexible text cleaner with configurable options."""
    # Decode HTML entities first (&amp; → &, &lt; → <)
    text = html.unescape(text)

    if strip_html:
        text = re.sub(r'<[^>]+>', ' ', text)  # remove HTML tags

    if strip_urls:
        # Matches http(s), www, and bare domain patterns
        text = re.sub(
            r'(https?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.(com|org|net|edu|gov)\S*)',
            ' ',
            text
        )

    if strip_emojis:
        # Unicode emoji range
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002600-\U000027BF"  # misc symbols
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub(' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

---

## Tokenization Comparison

### Method Selection Guide

| Tokenizer | Speed | Vocab | Languages | Best For |
|---|---|---|---|---|
| Whitespace | Fastest | Large | English | Profiling, simple baselines |
| NLTK word_tokenize | Fast | Large | Multilingual | Research baselines, punctuation handling |
| spaCy | Medium | Custom | 50+ | Full NLP pipelines |
| HuggingFace (BPE/WordPiece) | Medium | Fixed (30k-50k) | Multilingual | Transformer models |
| Character-level | Fast | ~100 | Any | Noisy text, morphologically rich languages |

### Whitespace Tokenization

```python
tokens = text.split()  # simplest possible
```

### NLTK word_tokenize

```python
import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import word_tokenize

tokens = word_tokenize("Dr. Smith couldn't attend the meeting.")
# ['Dr.', 'Smith', 'could', "n't", 'attend', 'the', 'meeting', '.']
```

### spaCy

```python
# pip install spacy && python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # disable unused components

doc = nlp("Dr. Smith couldn't attend the meeting.")
tokens = [token.text for token in doc]
lemmas = [token.lemma_ for token in doc]
pos_tags = [(token.text, token.pos_) for token in doc]
```

### HuggingFace AutoTokenizer (Subword: BPE / WordPiece)

```python
# pip install transformers
from transformers import AutoTokenizer

# WordPiece (BERT)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoded = bert_tokenizer(
    "Dr. Smith couldn't attend the meeting.",
    truncation=True,
    max_length=512,
    padding="max_length",
    return_tensors="pt"
)
print(bert_tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))

# BPE (RoBERTa, GPT-2)
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
encoded = roberta_tokenizer("Dr. Smith couldn't attend the meeting.", return_tensors="pt")

# Batch tokenization for datasets
def tokenize_batch(texts, tokenizer, max_length=256):
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt"
    )
```

### Character-Level Tokenization

```python
# Useful for noisy social media text, code-switching, or morphologically rich languages
def char_tokenize(text, max_len=1000):
    chars = list(text[:max_len])
    vocab = {c: i+1 for i, c in enumerate(sorted(set(chars)))}
    return [vocab.get(c, 0) for c in chars]
```

---

## Embedding Selection Decision Tree

```
What is your task and dataset size?
│
├─ Small dataset (<10k samples) + interpretability needed
│  └─ TF-IDF → sklearn TfidfVectorizer
│     + sparse, fast, interpretable feature weights
│
├─ Semantic similarity + enough labeled data (>50k)
│  └─ Word2Vec or GloVe → gensim
│     + dense embeddings, good for similarity tasks
│
├─ Modern NLP task (similarity, clustering, retrieval)
│  └─ Sentence Transformers → sentence-transformers library
│     + purpose-built for sentence-level semantics
│
└─ State-of-the-art + fine-tuning budget available
   └─ HuggingFace Transformers (BERT, RoBERTa, DeBERTa)
      + best performance, especially for classification/NER/QA
```

### TF-IDF (sklearn)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),      # unigrams and bigrams
    min_df=2,                 # ignore terms in < 2 docs
    max_df=0.95,              # ignore terms in > 95% of docs
    sublinear_tf=True         # apply log normalization to TF
)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Feature inspection
feature_names = vectorizer.get_feature_names_out()
# Top features for document 0
doc_features = X_train[0].toarray()[0]
top_indices = doc_features.argsort()[-10:][::-1]
print([feature_names[i] for i in top_indices])
```

### Word2Vec / GloVe (gensim)

```python
# pip install gensim
from gensim.models import Word2Vec, KeyedVectors

# Train Word2Vec
tokenized_corpus = [text.split() for text in train_texts]
model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=300,
    window=5,
    min_count=5,
    workers=4,
    epochs=10
)

# Load pretrained GloVe
# Download: https://nlp.stanford.edu/projects/glove/
glove = KeyedVectors.load_word2vec_format('glove.6B.300d.txt', binary=False, no_header=True)

# Document embedding: mean of word vectors
import numpy as np

def doc_embedding(text, model, vector_size=300):
    tokens = text.split()
    vecs = [model.wv[tok] for tok in tokens if tok in model.wv]
    if not vecs:
        return np.zeros(vector_size)
    return np.mean(vecs, axis=0)

doc_vecs = np.array([doc_embedding(t, model) for t in texts])
```

### Sentence Transformers

```python
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# 'all-MiniLM-L6-v2': fast, good quality
# 'all-mpnet-base-v2': best quality
# 'paraphrase-multilingual-MiniLM-L12-v2': multilingual
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True  # for cosine similarity via dot product
)
print(f"Embedding shape: {embeddings.shape}")  # (n_docs, 384)

# Semantic similarity
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings[:5])
```

### HuggingFace Transformers (BERT, RoBERTa, DeBERTa)

```python
from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")

def get_cls_embeddings(texts, tokenizer, model, batch_size=32):
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, truncation=True, max_length=512,
                                padding=True, return_tensors="pt")
            outputs = model(**encoded)
            # CLS token embedding
            cls_emb = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_emb.cpu().numpy())
    return np.vstack(all_embeddings)

embeddings = get_cls_embeddings(texts, tokenizer, model)
```

---

## NLP Evaluation Metrics

### BLEU (Machine Translation / Text Generation)

BLEU measures n-gram precision between hypothesis and one or more references. Includes a brevity penalty for short outputs.

```python
# pip install sacrebleu
import sacrebleu

# Single sentence
hypothesis = "The cat sat on the mat ."
references = [["The cat is on the mat ."]]

score = sacrebleu.sentence_bleu(hypothesis, references)
print(f"Sentence BLEU: {score.score:.2f}")

# Corpus-level (preferred for reporting)
hypotheses = ["The cat sat on the mat .", "It is a nice day today ."]
references_list = [
    ["The cat is on the mat .", "There is a nice day today ."]
]

corpus_score = sacrebleu.corpus_bleu(hypotheses, references_list)
print(f"Corpus BLEU: {corpus_score.score:.2f}")
# Always report tokenization + version: BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.5.1 = 35.2
print(corpus_score)
```

**Interpret**: BLEU > 40 is often considered good for MT; but always compare against the same baseline. BLEU is not reliable for short texts.

### ROUGE (Summarization)

ROUGE measures recall-oriented n-gram overlap. ROUGE-L uses longest common subsequence.

```python
# pip install rouge-score
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)

hypothesis = "The cat sat on a mat near the window."
reference = "A cat sat on the mat by the window."

scores = scorer.score(reference, hypothesis)
for metric, score in scores.items():
    print(f"{metric}: P={score.precision:.3f} R={score.recall:.3f} F1={score.fmeasure:.3f}")

# Corpus-level averaging
def corpus_rouge(hypotheses, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    totals = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        for k in totals:
            totals[k].append(scores[k].fmeasure)
    return {k: sum(v)/len(v) for k, v in totals.items()}

avg_scores = corpus_rouge(hypotheses, references_list[0])
print(avg_scores)
```

### BERTScore (Semantic Similarity)

BERTScore computes token-level similarity using BERT embeddings. More robust to paraphrases than BLEU/ROUGE.

```python
# pip install bert-score
from bert_score import score as bert_score

hypotheses = ["The cat sat on the mat.", "It is warm today."]
references = ["The cat is on the mat.", "Today is a warm day."]

P, R, F1 = bert_score(
    hypotheses,
    references,
    lang="en",
    model_type="roberta-large",  # or "bert-base-uncased" for speed
    verbose=True
)

print(f"BERTScore P: {P.mean():.4f}")
print(f"BERTScore R: {R.mean():.4f}")
print(f"BERTScore F1: {F1.mean():.4f}")

# Per-example scores
for hyp, ref, f in zip(hypotheses, references, F1):
    print(f"F1={f:.3f}  |  {hyp[:40]}  →  {ref[:40]}")
```

### Perplexity (Language Model Quality)

Lower perplexity = better language model fit. Defined as exp(cross-entropy loss).

```python
# Using HuggingFace for perplexity on a language model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

def compute_perplexity(text, model_name="gpt2", max_length=1024):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    encodings = tokenizer(text, return_tensors="pt")
    max_length = min(max_length, model.config.max_position_embeddings)
    input_ids = encodings.input_ids[:, :max_length]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # cross-entropy

    perplexity = math.exp(loss.item())
    return perplexity

ppl = compute_perplexity("The model learns to predict the next word in a sentence.")
print(f"Perplexity: {ppl:.2f}")
```

### Exact Match and F1 (QA Tasks)

```python
import re
import string
import collections

def normalize_answer(s):
    """Lower, remove punctuation, articles, extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def token_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

# Corpus-level
predictions = ["Paris", "The Treaty of Versailles"]
ground_truths = ["Paris", "Treaty of Versailles"]

em_scores = [exact_match(p, g) for p, g in zip(predictions, ground_truths)]
f1_scores = [token_f1(p, g) for p, g in zip(predictions, ground_truths)]
print(f"Corpus EM: {sum(em_scores)/len(em_scores):.3f}")
print(f"Corpus F1: {sum(f1_scores)/len(f1_scores):.3f}")
```

### When to Use Human Evaluation

Automatic metrics are insufficient when:
- Task involves creativity, coherence, or factuality (hallucination detection)
- BLEU/ROUGE scores are close between systems (statistical ties)
- Novel generation domains with no clear reference distribution
- Evaluation of open-ended dialogue or story generation

Human evaluation dimensions: fluency, adequacy/faithfulness, coherence, relevance, factual accuracy.

---

## Standard NLP Benchmarks by Task

| Task | Benchmark | Metric | Notes |
|---|---|---|---|
| Text classification | GLUE / SuperGLUE | Accuracy, F1, MCC | SST-2, MNLI, QQP, etc. |
| NER | CoNLL-2003 | Span-level F1 | English news; also OntoNotes |
| Machine translation | WMT (news) | BLEU (sacrebleu) | Many language pairs |
| Summarization | CNN/DailyMail | ROUGE-1/2/L | Abstractive preferred |
| QA (extractive) | SQuAD 1.1 / 2.0 | EM, F1 | 2.0 includes unanswerable Qs |
| QA (open domain) | Natural Questions | EM | Requires retrieval |
| Language modeling | Penn Treebank | Perplexity | Classic; Wikitext-103 preferred now |
| Coreference | OntoNotes | CoNLL F1 | |
| Relation extraction | TACRED | Micro-F1 | |
| Semantic similarity | STS-Benchmark | Pearson/Spearman r | |

---

## Data Augmentation for NLP

### Easy Data Augmentation (EDA)

```python
# pip install nltk
import random
import nltk
from nltk.corpus import wordnet

nltk.download(['wordnet', 'omw-1.4'], quiet=True)

def synonym_replacement(words, n=1):
    """Replace n random non-stopword words with a synonym."""
    new_words = words.copy()
    replaceable = [w for w in words if wordnet.synsets(w)]
    random.shuffle(replaceable)
    for word in replaceable[:n]:
        syns = wordnet.synsets(word)
        synonyms = [lemma.name() for syn in syns for lemma in syn.lemmas()
                    if lemma.name() != word]
        if synonyms:
            idx = new_words.index(word)
            new_words[idx] = random.choice(synonyms).replace('_', ' ')
    return new_words

def random_insertion(words, n=1):
    """Insert n random synonyms at random positions."""
    new_words = words.copy()
    for _ in range(n):
        all_syns = []
        for word in words:
            syns = wordnet.synsets(word)
            for syn in syns:
                for lemma in syn.lemmas():
                    all_syns.append(lemma.name())
        if all_syns:
            new_words.insert(random.randint(0, len(new_words)), random.choice(all_syns))
    return new_words

def random_deletion(words, p=0.1):
    """Randomly delete words with probability p."""
    if len(words) == 1:
        return words
    return [w for w in words if random.random() > p] or [random.choice(words)]

def random_swap(words, n=1):
    """Randomly swap n pairs of words."""
    new_words = words.copy()
    for _ in range(n):
        if len(new_words) >= 2:
            i, j = random.sample(range(len(new_words)), 2)
            new_words[i], new_words[j] = new_words[j], new_words[i]
    return new_words

def eda_augment(text, n_aug=4, alpha=0.1):
    """Generate n_aug augmented versions of text using EDA."""
    words = text.split()
    n = max(1, int(alpha * len(words)))
    augmented = []
    for _ in range(n_aug):
        choice = random.choice(['sr', 'ri', 'rd', 'rs'])
        if choice == 'sr':
            new_words = synonym_replacement(words, n)
        elif choice == 'ri':
            new_words = random_insertion(words, n)
        elif choice == 'rd':
            new_words = random_deletion(words)
        else:
            new_words = random_swap(words, n)
        augmented.append(' '.join(new_words))
    return augmented
```

### Back-Translation

```python
# Requires HuggingFace transformers and MarianMT models
# pip install transformers sentencepiece
from transformers import MarianMTModel, MarianTokenizer

def back_translate(texts, src_lang="en", pivot_lang="de"):
    """Translate to pivot language and back for augmentation."""
    fwd_model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{pivot_lang}"
    bwd_model_name = f"Helsinki-NLP/opus-mt-{pivot_lang}-{src_lang}"

    fwd_tokenizer = MarianTokenizer.from_pretrained(fwd_model_name)
    fwd_model = MarianMTModel.from_pretrained(fwd_model_name)
    bwd_tokenizer = MarianTokenizer.from_pretrained(bwd_model_name)
    bwd_model = MarianMTModel.from_pretrained(bwd_model_name)

    def translate(texts, tokenizer, model):
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**encoded, num_beams=4)
        return tokenizer.batch_decode(translated, skip_special_tokens=True)

    pivot = translate(texts, fwd_tokenizer, fwd_model)
    back = translate(pivot, bwd_tokenizer, bwd_model)
    return back

# Example
texts = ["The conference was very productive and informative."]
augmented = back_translate(texts, src_lang="en", pivot_lang="de")
print(augmented)
```

---

## Best Practices

1. **Profile before modeling** — always run `scripts/text_profiler.py` on a new dataset first.
2. **Match preprocessing to model** — uncased BERT expects lowercased text; GPT-2 does not want stopwords removed.
3. **Use sacrebleu for BLEU** — always report tokenization settings and version for reproducibility.
4. **Prefer corpus-level metrics** — sentence-level metrics have high variance.
5. **BERTScore > BLEU** for semantic evaluation — especially when paraphrasing is acceptable.
6. **Do not use perplexity cross-model** — perplexity is only comparable within the same vocabulary.
7. **Report all metrics** — BLEU and ROUGE often tell different stories; report both.
8. **Stratified splits** — ensure label distribution is maintained across train/val/test.
9. **De-duplicate before splitting** — near-duplicate texts cause test set leakage.
10. **Check OOV rate** before fine-tuning — high OOV means the pre-trained vocab may not cover your domain.
