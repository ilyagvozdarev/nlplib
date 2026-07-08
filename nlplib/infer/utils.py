def gen_batch(records, batch_size):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start: batch_end]
        batch_start = batch_end
        yield batch


def conversations_to_inputs_prompts(
    tokenizer, 
    conversations, 
    tokenizer_params,
    device='cuda'
):
    assert isinstance(conversations[-1], list), 'conversation not a list'

    prompts = tokenizer.apply_chat_template(
        conversations, 
        tokenize=False, 
        **tokenizer_params
    )

    inputs = []
    for prompt in prompts:
        input = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        input.pop("token_type_ids", None)
        inputs.append(input)

    return inputs, prompts
