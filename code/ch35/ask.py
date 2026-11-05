# Ask the book some questions, including two it cannot answer.
from qa import BookQA

qa = BookQA()
for q in ["What is the vanishing gradient problem?",
          "Why use a validation set?",
          "How do I bake sourdough bread?",
          "What is the capital of Australia?"]:
    r = qa.answer(q)
    print(f"Q: {q}\nA: {r['answer']}\n   sources: {'; '.join(r['sources']) or '-'}  (match {r['confidence']})\n")
