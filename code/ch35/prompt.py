# Build a grounded prompt for a language model from the retrieved passages.
from qa import BookQA

SYSTEM = ("You answer questions about the book 'Wired to Learn'. Use ONLY the numbered passages provided. "
          "Cite the passages you use like [1]. If the passages do not contain the answer, say "
          "'The book does not cover this.' Do not use outside knowledge.")


def build_prompt(qa, question, k=3):
    top, _ = qa.retrieve(question, k=k)
    context = "\n\n".join(f"[{n}] (Chapter {qa.P[i]['chapter']}, \"{qa.P[i]['section']}\")\n{qa.P[i]['text']}"
                          for n, i in enumerate(top, 1))
    return f"Passages:\n\n{context}\n\nQuestion: {question}"


if __name__ == "__main__":
    prompt = build_prompt(BookQA(), "What is the capital of Australia?")
    print(f"SYSTEM: {SYSTEM}\n\nUSER: {prompt[:520]}...\n[{len(prompt.split())} words in total]")
