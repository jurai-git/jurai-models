from transformers import T5Tokenizer, T5ForConditionalGeneration
from models.arguments.model_utils import summarize, show_args

tokenizer = T5Tokenizer.from_pretrained('./t5_finetuned_tokenizer')
model = T5ForConditionalGeneration.from_pretrained('./t5_finetuned_model')

text = input()

summaries = summarize(text, model, tokenizer)
show_args(summaries)
