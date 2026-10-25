# URL-Based RAG Assistant

[![tests](https://github.com/sarmadmirza8888/url-rag-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/sarmadmirza8888/url-rag-assistant/actions)

Paste a webpage URL, ask a question, and get an answer grounded in that page's content.
A Retrieval-Augmented Generation (RAG) pipeline built with a **locally run Mistral 7B**,
PostgreSQL + **pgvector**, and a Gradio interface.

**Sample output:**![sample png](/docs/images/sample.png)
[sample2 png](/docs/images/sample2.png)


## Why it exists

LLMs don't know what is on a specific page today, can't fit long pages in their context window,
and may hallucinate. RAG retrieves only the relevant passages and forces the model to answer from
them, or say it couldn't find the answer.

## How it works

```
URL -> validate -> fetch -> extract -> chunk -> embed -> pgvector
Question -> embed -> similarity search (top 3) -> grounded prompt -> Mistral 7B -> answer
```

Full diagram and design decisions: [docs/architecture.md](docs/architecture.md)

## Highlights

- **Security-minded fetching (SSRF protection):** public http(s) only, DNS resolved and checked on every redirect, size and content-type limits
- **Grounded answers:** system prompt restricts the model to retrieved context and treats page text as data, not instructions
- **Vector search in SQL:** cosine similarity with pgvector
- **Tested:** automated tests for validation, chunking, fetching (including a Wikipedia-style HTTP 403 scenario) and the pipeline, run in CI
- **Local-first:** the LLM runs on your own GPU, no paid API required

## Project structure

```
url-rag-assistant/
├── app.py                  # Gradio interface
├── src/url_rag/            # application code (see docs/architecture.md)
├── tests/                  # pytest suite
├── notebooks/              # original Kaggle notebook (step-by-step walkthrough)
├── db/schema.sql           # PostgreSQL + pgvector table
├── docs/                   # architecture notes and screenshots
├── .github/workflows/      # CI
├── .env.example            # configuration template
└── requirements*.txt
```

## Run the tests (no GPU needed)

```bash
pip install -r requirements-dev.txt
pytest
```

## Run the app

Requires a CUDA GPU (developed on a Kaggle T4) and a PostgreSQL database with pgvector.

```bash
pip install -r requirements.txt
psql "$DATABASE_URL" -f db/schema.sql      # create the table once
cp .env.example .env                       # fill in your values, then export them
python app.py
```

Or open `notebooks/url_rag_assistant.ipynb` on Kaggle and run it top to bottom
(add `DATABASE_URL` under Add-ons > Secrets).

## Limitations

- Retrieves only 3 chunks of ~800 characters, so broad questions on very long pages get partial answers
- JavaScript-rendered pages are not supported
- Character-based chunking; no re-ranking or evaluation set yet

## Roadmap

- [ ] Show the retrieved source chunks next to each answer
- [ ] Small evaluation set (retrieval hit rate)
- [ ] Hosted-LLM option for a free public demo

## Author

**Muhammad Sarmad Mirza** · www.linkedin.com/in/muhammad-sarmad-mirza-1a1a66381 · sarmadmirza8888@gmail.com

Licensed under the MIT License.
