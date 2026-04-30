import os



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
    '''
        преобразует разговоры в промпты (применяя шаблон чата)
        и тензоры кодированных входов на заданном device

        Returns:
            prompts - промпт
            inputs - кодированный вход
    '''

    print('\nutils.conversations_to_inputs_prompts\n')
    print(f'tokenizer_params = {tokenizer_params}')

    assert isinstance(conversations[-1], list), 'conversation not a list'

    prompts = tokenizer.apply_chat_template(
        conversations, 
        tokenize=False, 
        **tokenizer_params
    )

    print(f'prompts count = {len(prompts)}')

    # print(prompts[:70000])

    inputs = []

    for prompt in prompts:
        # add_special_tokens=False тк спец токены добавились в apply_chat_template
        input = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        # для старых версий transformers:
        # input = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        # input = {k: v.to(device) for k, v in input.items()}

        input.pop("token_type_ids", None)
        inputs.append(input)
    
    print('inputs device = ', list(inputs[0].values())[0].device)

    return inputs, prompts
