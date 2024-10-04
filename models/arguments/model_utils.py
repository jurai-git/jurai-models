def show_args(summaries):
    for idx, text in enumerate(summaries):
        for arg in text:
            print(f'Argument {idx + 1}: {arg}')
        print()


def summarize(text, model, tokenizer, input_text='return the best arguments: ', max_length=1028, num_outputs=3):
    input_text += text
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=max_length, truncation=True)

    summary_ids = model.generate(
        input_ids,
        max_length=256,
        min_length=48,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
        num_return_sequences=num_outputs
    )
    return [tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=True) for i in summary_ids]
