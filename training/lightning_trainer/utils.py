from model import tokenizer


def preprocess_function(sample, max_source_length, max_target_length):
    templated_text, labels = formatting(sample, max_source_length, max_target_length)

    return {
        "input_ids": templated_text,
        "labels": labels
    }

def formatting(sample,
               max_source_length,
               max_target_length,
               padding="max_length"):
    # add prefix to the input for t5
    model_inputs, labels = [], []
    for dialogue, summary in zip(sample["dialogue"], sample["summary"]):
        chat_template = [
            # {
            #     "role": "system",
            #     "content": "You are a friendly chatbot who always responds with summary",
            # },
            {
                "role": "user",
                "content": f"Summarize the following dialogue\n\n{dialogue}"
            },

        ]
        label_template = [{
                "role": "assistant",
                "content": f"{summary}"
        }]

        chat_message = tokenizer.apply_chat_template(conversation=chat_template,
                                                     tokenize=False,
                                                     add_generateion_prompt=False, )
        bot_message = tokenizer.apply_chat_template(conversation=label_template,
                                                    tokenize=False,
                                                    add_generateion_prompt=False, )
        model_inputs.append(chat_message)
        labels.append(bot_message)

    # inputs = ["summarize: " + item for item in sample["dialogue"]]

    # tokenize inputs
    # model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True, )

    # Tokenize targets with the `text_target` keyword argument
    # labels = tokenizer(text_target=sample["summary"],
    #                    max_length=max_target_length,
    #                    padding=padding,
    #                    truncation=True,)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    # if padding == "max_length":
    #     labels["input_ids"] = [
    #         [(l if l != tokenizer.pad_token_id else 1) for l in label] for label in labels["input_ids"]
    #     ]
    # model_inputs["labels"] = labels["input_ids"]
    return model_inputs, labels