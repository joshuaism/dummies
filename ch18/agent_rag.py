from google import genai
import math
import re
from collections import Counter
import os

# get api key from https://aistudio.google.com
gemini_api_key = os.environ.get('GEMINI_KEY')

# create client
client = genai.Client(api_key=gemini_api_key)

# --- Tiny document store ---
DOCS = [
    "RAG stands for Retrieval-Augmented Generation.",
    "Agentic RAG allows an agent to plan, retrieve multiple times, and refine its queries.",
    "RAG reduces hallucinations by grounding generation in retrieved documents.",
    "Dogs love RAG more than Cats"
]

# --- Simple retriever (bag-of-words) ---
WORD_RE = re.compile(r"[a-z0-9]+")


def bow(text):
    return Counter(WORD_RE.findall(text.lower()))


def cosine(a, b):
    dot = sum(a[t] * b.get(t, 0) for t in a)
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(query, k=2):
    """uses query to retrieve relevant and sorted clauses from DOCS based on score"""
    qv = bow(query)
    scored = [(cosine(qv, bow(d)), d) for d in DOCS]
    scored.sort(reverse=True, key=lambda x: x[0])
    return [d for s, d in scored[:k] if s > 0]

# --- LLM call (generation) ---


def generate(question, context):
    prompt = f"""Use the context to answer.
Context: {context}

Question: {question}
"""

    response = client.models.generate_content(
        model="models/gemini-3.5-flash-lite",
        contents=prompt
    )
    return response.text

# --- Aagentic RAG loop ---


def agentic_rag(question, max_steps=2):
    query = question

    for step in range(max_steps):
        docs = retrieve(query)
        context = "\n".join(docs)
        print(context)

        answer = generate(question, context)

        # crude "agent decision"
        if "stand for" in answer.lower() or len(docs) >= 2:
            return answer + f"\n\n(Answered in {step+1} step(s))"

        # agent refines the query
        query = question + " definition meaning"

    return answer + "\n\n(Stopped after max steps)"


# --- Run ---
if __name__ == "__main__":
    q = "What does RAG stand for in Agentic RAG?"
    print(agentic_rag(q))
    q = "Do Dogs love RAG?"
    print(agentic_rag(q))
    q = "Do Cats love RAG?"
    print(agentic_rag(q))
