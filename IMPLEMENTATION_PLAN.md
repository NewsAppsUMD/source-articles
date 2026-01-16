# Implementation Plan: Hybrid Fine-Tuning + RAG System for OpenNews Articles

## Overview

Build a multi-purpose journalism assistant using OpenNews Source articles (1,183 articles, ~1M tokens) with a hybrid architecture combining:
- **Fine-tuned local model** (Qwen2.5-14B or Mistral-Nemo-12B) for internalized journalism knowledge
- **RAG system** (ChromaDB + sentence-transformers) for accurate article retrieval and citations
- **Smart router** to select the best approach for each query

## System Architecture

```
User Query
    ↓
Intent Router (rule-based)
    ↓
    ├─→ Fine-tuned Model (general knowledge, generation)
    │   └─→ Fast response without retrieval
    │
    └─→ RAG System (specific articles, citations)
        └─→ Retrieve + Generate with sources
```

## Technology Stack

### Core Dependencies
```
transformers>=4.36.0
peft>=0.7.0
bitsandbytes>=0.41.0
torch>=2.1.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
langchain>=0.1.0
accelerate>=0.25.0
datasets>=2.16.0
```

### Hardware Requirements
- GPU: 24GB+ VRAM (tested on RTX 4090, A6000, etc.)
- Disk: ~100GB free space
- RAM: 32GB+ recommended

## Step-by-Step Implementation

---

### Phase 1: Environment Setup (Day 1)

**Goal:** Prepare development environment and download base model

**Steps:**

1. **Create project structure**
```bash
mkdir -p data models/fine_tuned models/base vector_db scripts notebooks
```

2. **Install dependencies**
```bash
uv add transformers peft bitsandbytes torch chromadb sentence-transformers langchain accelerate datasets
```

3. **Download base model**

Create `scripts/0_download_model.py`:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-14B-Instruct"  # or "mistralai/Mistral-Nemo-Instruct-2407"

print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)

tokenizer.save_pretrained("./models/base")
model.save_pretrained("./models/base")
print("Model downloaded successfully!")
```

Run: `uv run python scripts/0_download_model.py`

4. **Verify GPU**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

**Validation:** Model downloaded, GPU detected, memory sufficient

---

### Phase 2: Build RAG System (Days 2-3)

**Goal:** Create vector database and retrieval pipeline (baseline system)

#### Step 2.1: Chunk Articles

Create `scripts/2_prepare_rag_chunks.py`:
```python
import json
from typing import List, Dict

def chunk_article(article: Dict, chunk_size: int = 800, overlap: int = 100) -> List[Dict]:
    """Split article text into overlapping chunks."""
    text = article['text']
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)

        chunks.append({
            'text': chunk_text,
            'metadata': {
                'url': article['url'],
                'title': article['title'],
                'authors': article['authors'],
                'date': article['date'],
                'chunk_id': f"{article['url']}#chunk_{len(chunks)}"
            }
        })

    return chunks

# Load and chunk all articles
with open('articles.json', 'r') as f:
    articles = json.load(f)

all_chunks = []
for article in articles:
    chunks = chunk_article(article)
    all_chunks.extend(chunks)

print(f"Created {len(all_chunks)} chunks from {len(articles)} articles")

# Save chunks
with open('data/chunks.jsonl', 'w') as f:
    for chunk in all_chunks:
        f.write(json.dumps(chunk) + '\n')
```

Run: `uv run python scripts/2_prepare_rag_chunks.py`

#### Step 2.2: Build Vector Database

Create `scripts/4_setup_rag.py`:
```python
import json
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./vector_db")

# Use sentence-transformers for embeddings
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create or get collection
collection = client.get_or_create_collection(
    name="opennews_articles",
    embedding_function=sentence_transformer_ef,
    metadata={"description": "OpenNews Source articles chunked for RAG"}
)

# Load chunks
chunks = []
with open('data/chunks.jsonl', 'r') as f:
    chunks = [json.loads(line) for line in f]

print(f"Loading {len(chunks)} chunks into ChromaDB...")

# Add in batches (ChromaDB has batch size limits)
batch_size = 500
for i in tqdm(range(0, len(chunks), batch_size)):
    batch = chunks[i:i + batch_size]

    collection.add(
        documents=[c['text'] for c in batch],
        metadatas=[c['metadata'] for c in batch],
        ids=[c['metadata']['chunk_id'] for c in batch]
    )

print(f"Vector database created with {collection.count()} chunks")
```

Run: `uv run python scripts/4_setup_rag.py`

#### Step 2.3: Test Retrieval

Create `scripts/test_rag.py`:
```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./vector_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_collection(
    name="opennews_articles",
    embedding_function=sentence_transformer_ef
)

# Test query
query = "What are best practices for peer support in journalism?"
results = collection.query(
    query_texts=[query],
    n_results=5
)

print(f"Query: {query}\n")
print("Top 5 Results:")
for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"\n{i+1}. {meta['title']}")
    print(f"   URL: {meta['url']}")
    print(f"   Text: {doc[:200]}...")
```

Run: `uv run python scripts/test_rag.py`

**Validation:** Retrieval returns relevant chunks with metadata

---

### Phase 3: Generate Training Data (Days 3-4)

**Goal:** Create instruction-following Q&A pairs for fine-tuning

Create `scripts/1_prepare_training_data.py`:
```python
import json
import os
from anthropic import Anthropic  # or use OpenAI

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a training data generator. Given a journalism article, create 5-7 diverse question-answer pairs that would help train a journalism assistant model.

Create questions of different types:
1. Factual questions (who, what, when, where)
2. Conceptual questions (why, how, explain)
3. Application questions (how to implement X)
4. Comparison questions (compare X and Y)
5. Summary questions (summarize the main points)

Format each as:
Q: [question]
A: [detailed answer based on article content]

Do not include citations or references to "this article" - answer as if the knowledge is internalized."""

def generate_qa_pairs(article: dict) -> list:
    """Generate Q&A pairs for an article using Claude."""

    prompt = f"""Article Title: {article['title']}
Authors: {', '.join(article['authors'])}
Date: {article['date']}

Article Text:
{article['text'][:4000]}  # Truncate if too long

Generate 5-7 question-answer pairs:"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse response into Q&A pairs
    qa_pairs = []
    text = response.content[0].text
    pairs = text.split('\n\n')

    for pair in pairs:
        if 'Q:' in pair and 'A:' in pair:
            q_part = pair.split('A:')[0].replace('Q:', '').strip()
            a_part = pair.split('A:')[1].strip()

            qa_pairs.append({
                "instruction": q_part,
                "input": "",
                "output": a_part,
                "source_article": article['url']
            })

    return qa_pairs

# Load articles
with open('articles.json', 'r') as f:
    articles = json.load(f)

# Generate training data (with rate limiting)
all_training_data = []
import time

for i, article in enumerate(articles):
    print(f"Processing article {i+1}/{len(articles)}: {article['title']}")

    try:
        qa_pairs = generate_qa_pairs(article)
        all_training_data.extend(qa_pairs)
        print(f"  Generated {len(qa_pairs)} Q&A pairs")
    except Exception as e:
        print(f"  Error: {e}")

    # Rate limiting
    time.sleep(1)

    # Save progress every 50 articles
    if (i + 1) % 50 == 0:
        with open('data/training_data.jsonl', 'w') as f:
            for item in all_training_data:
                f.write(json.dumps(item) + '\n')

# Final save
with open('data/training_data.jsonl', 'w') as f:
    for item in all_training_data:
        f.write(json.dumps(item) + '\n')

print(f"\nTotal training examples: {len(all_training_data)}")
```

**Note:** This will use Claude API and cost approximately $20-50 depending on article lengths.

Alternative: Use open-source models (llama.cpp, Ollama) locally for free but lower quality.

Run: `uv run python scripts/1_prepare_training_data.py`

**Validation:** Generated 6,000-12,000 Q&A pairs, manually review 50 random samples for quality

---

### Phase 4: Fine-Tune Model (Days 5-6)

**Goal:** Train model using QLoRA on generated Q&A pairs

Create `scripts/3_finetune_model.py`:
```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import bitsandbytes as bnb

# Load model in 4-bit
model_name = "./models/base"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": torch.float16
    }
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Prepare for training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load and format dataset
dataset = load_dataset('json', data_files='data/training_data.jsonl', split='train')

def format_instruction(example):
    """Format as instruction-following prompt."""
    if example['input']:
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

    return tokenizer(text, truncation=True, max_length=2048, padding=False)

tokenized_dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)

# Split train/validation
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/fine_tuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    save_steps=500,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=3,
    warmup_steps=100,
    lr_scheduler_type="cosine"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset['train'],
    eval_dataset=split_dataset['test'],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Train
print("Starting training...")
trainer.train()

# Save final model
model.save_pretrained("./models/fine_tuned/final")
tokenizer.save_pretrained("./models/fine_tuned/final")
print("Training complete!")
```

Run: `uv run python scripts/3_finetune_model.py`

**Expected time:** 4-8 hours on 24GB GPU

**Validation:** Training loss decreases, validation perplexity improves, model generates coherent responses

---

### Phase 5: Build Router and Integration (Day 7)

**Goal:** Create unified inference system with smart routing

Create `scripts/5_inference.py`:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import chromadb
from chromadb.utils import embedding_functions

class JournalismAssistant:
    def __init__(self):
        # Load fine-tuned model
        print("Loading fine-tuned model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "./models/base",
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model = PeftModel.from_pretrained(base_model, "./models/fine_tuned/final")
        self.tokenizer = AutoTokenizer.from_pretrained("./models/base")

        # Load RAG components
        print("Loading RAG system...")
        client = chromadb.PersistentClient(path="./vector_db")
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = client.get_collection(
            name="opennews_articles",
            embedding_function=sentence_transformer_ef
        )

        print("System ready!")

    def route_query(self, query: str) -> str:
        """Determine which system to use."""
        citation_keywords = ["which articles", "find articles", "source", "reference", "who wrote", "list articles"]
        factual_keywords = ["when did", "what year", "what date", "specific", "exact"]

        query_lower = query.lower()

        if any(kw in query_lower for kw in citation_keywords):
            return "rag"
        elif any(kw in query_lower for kw in factual_keywords):
            return "rag"
        elif "?" in query and len(query.split()) < 15:
            return "rag"  # Short factual questions
        else:
            return "fine_tuned"

    def generate_with_finetuned(self, query: str) -> str:
        """Generate response using fine-tuned model."""
        prompt = f"### Instruction:\n{query}\n\n### Response:\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response part
        response = response.split("### Response:\n")[-1].strip()

        return response

    def generate_with_rag(self, query: str, top_k: int = 5) -> dict:
        """Generate response using RAG."""
        # Retrieve relevant chunks
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        # Format context
        context = "\n\n---\n\n".join([
            f"[{meta['title']}]({meta['url']})\n{doc}"
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ])

        # Generate with context
        prompt = f"""### Instruction:
Answer the following question based on the provided sources. Include citations.

Question: {query}

Sources:
{context}

### Response:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### Response:\n")[-1].strip()

        return {
            "response": response,
            "sources": [
                {"title": meta['title'], "url": meta['url']}
                for meta in results['metadatas'][0]
            ]
        }

    def query(self, query: str) -> dict:
        """Main query interface."""
        route = self.route_query(query)
        print(f"Routing to: {route}")

        if route == "rag":
            result = self.generate_with_rag(query)
            return {
                "query": query,
                "route": "rag",
                "response": result['response'],
                "sources": result['sources']
            }
        else:
            response = self.generate_with_finetuned(query)
            return {
                "query": query,
                "route": "fine_tuned",
                "response": response,
                "sources": []
            }

# CLI Interface
if __name__ == "__main__":
    assistant = JournalismAssistant()

    print("\nJournalism Assistant Ready!")
    print("Type 'quit' to exit\n")

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ['quit', 'exit']:
            break

        result = assistant.query(query)
        print(f"\nAssistant ({result['route']}): {result['response']}")

        if result['sources']:
            print("\nSources:")
            for source in result['sources']:
                print(f"  - {source['title']}: {source['url']}")
```

Run: `uv run python scripts/5_inference.py`

**Validation:** System routes queries correctly, both modes generate quality responses

---

### Phase 6: Evaluation and Refinement (Days 8-10)

**Goal:** Measure performance and iterate

Create `notebooks/evaluation.ipynb`:
```python
# Test queries covering different types
test_queries = [
    # Factual/Retrieval (should route to RAG)
    "When was NICAR25 held?",
    "Which articles discuss peer support in journalism?",
    "Who wrote about AI at NICAR?",

    # Conceptual (should route to fine-tuned)
    "Explain the benefits of peer support networks for journalists",
    "What are the main challenges in data journalism?",
    "How has AI changed journalism workflows?",

    # Creative/Generation (should route to fine-tuned)
    "Write a summary of best practices for newsroom mental health",
    "Describe how to set up a peer support program",

    # Complex (test both)
    "Compare fine-tuning and RAG approaches for journalism applications"
]

# Evaluation metrics
from assistant import JournalismAssistant

assistant = JournalismAssistant()

results = []
for query in test_queries:
    result = assistant.query(query)
    results.append(result)
    print(f"\nQuery: {query}")
    print(f"Route: {result['route']}")
    print(f"Response: {result['response'][:200]}...")
    if result['sources']:
        print(f"Sources: {len(result['sources'])} articles")
```

**Evaluation Checklist:**
- [ ] Fine-tuned model generates coherent, journalism-focused responses
- [ ] RAG retrieves relevant articles with accurate citations
- [ ] Router makes appropriate decisions for different query types
- [ ] Response quality is good for both routes
- [ ] Latency is acceptable (<10 seconds)
- [ ] No hallucinated sources in RAG responses
- [ ] Model doesn't repeat "this article" or break character

**Iteration:**
- Adjust router keywords based on misrouted queries
- Tune top_k in RAG (try 3, 5, 10)
- Adjust temperature/top_p for generation quality
- Consider re-training if fine-tuned model underperforms
- Add re-ranking to RAG if retrieval quality is poor

---

## Success Criteria

✅ **RAG System:**
- Retrieves relevant articles with >80% precision
- Includes accurate citations in responses
- Handles "find articles about X" queries

✅ **Fine-tuned Model:**
- Generates journalism-domain responses
- No catastrophic forgetting (still answers general questions)
- Faster than RAG for general queries

✅ **Hybrid System:**
- Router accuracy >85% on test queries
- End-to-end latency <10 seconds
- User satisfaction with response quality

---

## Maintenance and Updates

**Adding New Articles:**
1. Scrape new articles with `scraper.py`
2. Chunk and add to vector DB: `scripts/2_prepare_rag_chunks.py`
3. Optionally: Generate new training data and re-fine-tune (quarterly)

**Improving Performance:**
- Collect user feedback on responses
- A/B test different routing strategies
- Experiment with different base models (Llama-3, Qwen-2.5, Mistral)
- Try different embedding models (OpenAI ada-002, Cohere, BGE)

---

## Cost Estimates

**One-Time:**
- Training data generation: $20-50 (Claude API)
- Fine-tuning compute: Free (local GPU)

**Ongoing:**
- Inference: Free (local)
- Storage: ~100GB disk space

**Time Investment:**
- Initial setup: ~10 days
- Maintenance: ~2-4 hours/month

---

## Troubleshooting

**Out of Memory during training:**
- Reduce batch size to 2
- Increase gradient_accumulation_steps to 8
- Use smaller LoRA rank (32 instead of 64)

**Poor retrieval quality:**
- Try different embedding model (all-mpnet-base-v2)
- Adjust chunk size (smaller = more precise, larger = more context)
- Add re-ranking with cross-encoder

**Model generates gibberish:**
- Check training data quality
- Reduce learning rate
- Train for fewer epochs (might be overfitting)

**Router makes wrong decisions:**
- Add more keyword patterns
- Log misrouted queries and adjust rules
- Consider training a small classifier (future enhancement)

---

## Next Steps

Ready to start implementation?

**Recommended order:**
1. Phase 1: Setup (quick win, verify hardware)
2. Phase 2: RAG (establish baseline, immediate results)
3. Phase 3: Generate training data (can run overnight)
4. Phase 4: Fine-tune (long-running, can monitor)
5. Phase 5: Integration (bring it all together)
6. Phase 6: Evaluate and iterate

Let me know when you're ready to begin, and I can help with any phase!
