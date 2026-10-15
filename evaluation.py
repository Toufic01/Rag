"""
evaluation.py  -  measure how good the system is.

    python evaluation.py            -> compare 2 embedding models x 4 chunking setups (no LLM needed)
    python evaluation.py answers    -> ask all questions end to end, check "not found", measure latency (needs Ollama)

Questions + the expected source/keyword are in questions.json.
A retrieved chunk counts as CORRECT if it comes from the expected file AND contains an expected keyword.
"""
import csv
import json
import sys
import time
import uuid
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

import rag_system as rag

QUESTIONS = json.load(open("questions.json", encoding="utf-8"))
MODELS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
CHUNKINGS = [("fixed", 300, 50), ("fixed", 500, 100), ("fixed", 1000, 200), ("sentence", 800, 0)]
K = 4
BGE_PREFIX = "Represent this sentence for searching relevant passages: "


def norm(text):                                   # ignore line breaks / capital letters when matching keywords
    return " ".join(text.split()).lower()


def is_correct(text, source, q):
    return source in q["sources"] and any(norm(k) in norm(text) for k in q["keywords"])


# ----------------------------------------------------------------- 1) retrieval comparison
def compare_retrieval():
    files = [p for p in sorted(Path("data").iterdir()) if p.suffix.lower() in rag.SUPPORTED]
    db = chromadb.EphemeralClient()               # temporary database in memory (does not touch chroma_db/)
    rows = []
    for model_name in MODELS:
        model = SentenceTransformer(model_name)
        prefix = BGE_PREFIX if "bge" in model_name else ""
        for strategy, size, overlap in CHUNKINGS:
            col = db.create_collection(f"t{uuid.uuid4().hex[:10]}", metadata={"hnsw:space": "cosine"})

            # --- index all documents with this setup (and time it)
            t0 = time.time()
            chunks, metas = [], []
            for f in files:
                for seg in rag.load_file(f):
                    for piece in rag.chunk_text(seg["text"], strategy, size, overlap):
                        chunks.append(piece)
                        metas.append({"source": f.name})
            col.add(ids=[str(i) for i in range(len(chunks))], documents=chunks, metadatas=metas,
                    embeddings=model.encode(chunks, normalize_embeddings=True).tolist())
            index_s = time.time() - t0

            # --- ask every question and score the retrieved chunks
            hit_at_1 = hit_at_k = 0
            rr, times, ans_dist, unans_dist = [], [], [], []
            for q in QUESTIONS:
                t = time.time()
                res = col.query(query_embeddings=model.encode([prefix + q["question"]], normalize_embeddings=True).tolist(),
                                n_results=min(K, len(chunks)))
                times.append(time.time() - t)
                top1 = res["distances"][0][0]
                if not q["answerable"]:
                    unans_dist.append(top1)       # distance of the best match for a question that has NO answer
                    continue
                ans_dist.append(top1)
                ranks = [i for i, (d, m) in enumerate(zip(res["documents"][0], res["metadatas"][0]), 1)
                         if is_correct(d, m["source"], q)]
                if ranks:
                    hit_at_k += 1
                    hit_at_1 += ranks[0] == 1
                rr.append(1 / ranks[0] if ranks else 0)
            n = len(ans_dist)
            rows.append({
                "model": model_name.split("/")[-1], "chunking": f"{strategy}-{size}-{overlap}", "chunks": len(chunks),
                "hit@1": round(hit_at_1 / n, 2), f"hit@{K}": round(hit_at_k / n, 2), "MRR": round(sum(rr) / n, 2),
                "index_s": round(index_s, 2), "retrieval_ms": round(1000 * sum(times) / len(times), 1),
                "dist_answerable": round(sum(ans_dist) / n, 2),          # should be LOW
                "dist_unanswerable": round(sum(unans_dist) / len(unans_dist), 2),   # should be HIGH
            })
            db.delete_collection(col.name)
            print("done:", rows[-1]["model"], rows[-1]["chunking"])

    print("\n" + " | ".join(rows[0]))
    for r in rows:
        print(" | ".join(str(v) for v in r.values()))
    write_csv("results_retrieval.csv", rows)
    print("\nTip: set MAX_DISTANCE in rag_system.py between dist_answerable and dist_unanswerable of your best setup.")


# ----------------------------------------------------------------- 2) end-to-end answers + latency
def evaluate_answers():
    print("Indexing sample documents with the settings from rag_system.py ...")
    rag.ingest_folder("data")
    rag.ask(QUESTIONS[0]["question"])             # warm-up (first call loads the LLM into memory, do not count it)
    rows = []
    for q in QUESTIONS:
        t = time.time()
        rag.retrieve(q["question"])               # time retrieval alone
        retrieval_s = time.time() - t
        answer, hits, total_s = rag.ask(q["question"])
        rows.append({
            "id": q["id"], "question": q["question"], "answerable": q["answerable"],
            "expected": q["expected_answer"] or "(not in documents)", "answer": answer.replace("\n", " "),
            "sources": "; ".join(sorted({h["source"] for h in hits})),
            "keyword_in_answer": any(norm(k) in norm(answer) for k in q["keywords"]) if q["answerable"] else "",
            "correct_refusal": (not hits) if not q["answerable"] else "",
            "retrieval_s": round(retrieval_s, 2), "total_s": round(total_s, 2),
            "manual_score": "",                   # fill in by hand: 0 wrong, 1 partly, 2 correct
        })
        print(f"[{q['id']:>2}] {total_s:5.1f}s  {q['question']}")

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    print("\n=== Summary ===")
    print(f"Keyword found in answer: {sum(r['keyword_in_answer'] is True for r in ans)}/{len(ans)}")
    print(f"Correct refusals       : {sum(r['correct_refusal'] is True for r in unans)}/{len(unans)}")
    totals = [r["total_s"] for r in rows]
    print(f"Latency (s)            : mean {sum(totals)/len(totals):.2f} | max {max(totals):.2f}")
    write_csv("results_answers.csv", rows)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("Saved", path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "answers":
        evaluate_answers()
    else:
        compare_retrieval()
