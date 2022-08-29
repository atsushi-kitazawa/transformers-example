from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained(
#     'cl-tohoku/bert-base-japanese-whole-word-masking')
# model = AutoModelForMaskedLM.from_pretrained(
#     'cl-tohoku/bert-base-japanese-whole-word-masking')
qa_model = pipeline(task='question-answering')
question = "Where do I live?"
context = "My name is Merve and I live in Japan."
ret = qa_model(question=question, context=context)
print(ret)
