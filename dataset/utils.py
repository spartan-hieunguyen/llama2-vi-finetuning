def tokenize(prompt, tokenizer, add_eos_token=False, max_len=512):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < max_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(
    data_point, prompter, tokenize_fn, train_on_inputs=True, add_eos_token=False
):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize_fn(full_prompt)
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize_fn(user_prompt, add_eos_token=add_eos_token)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
    return tokenized_full_prompt


def generate_prompt(example, prompter):
    text = prompter.generate_prompt(
        example["instruction"], example["input"], example["output"]
    )
    example["text"] = text
    return example
