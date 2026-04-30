import torch


@torch.inference_mode
def fix_untrained_tokens(model, tokenizer, eps: float = 1e-16) -> None:

    '''
        необученным токенам (с нулевым входным эмбеддингом) назначаем входной/выходной эмбеддинг равный среднему входному/выходному 
        эмбеддингу обученных токенов (кроме специальных)
  
	    1. находим необученные токены - индексы токенов словаря (кроме специальных) у которых нулевой вектор входного эмбеддинга 
           (наибольший элемент вектора меньше eps = 1e-16)
        2. находим средний входной/выходной эмбеддинг обученных токенов:
		   находим вектор - сумму векторов всех входных эмбеддингов и вектор - сумму векторов всех выходных и 
		   вычитаем из первого из них вектор-сумму всех входных эмбеддингов необученных токенов, а из второго вектор-сумму всех 
           выходных эмбеддингов необученных токенов и делим их на количество обученных токенов
	    3. необученным токенам назначаем средний входной/выходной эмбеддинг обученных токенов 
    '''

    embedding_matrix = model.get_input_embeddings().weight
    lm_head_matrix = model.get_output_embeddings().weight
    assert embedding_matrix.shape[0] == lm_head_matrix.shape[0]

    indicator_untrained = torch.amax(embedding_matrix, axis=1) <= eps
    special_tokens = (
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    )

    # исключаем специальные
    for special_token in special_tokens:
        if hasattr(tokenizer, special_token + "_id"):
            token_id = eval(f"tokenizer.{special_token}_id")
            if token_id is not None and token_id < indicator_untrained.shape[0]:
                indicator_untrained[token_id] = False

    where_untrained = torch.where(indicator_untrained)[0]       # индексы необученных токенов
    n_untrained = where_untrained.shape[0]
    n_trained = embedding_matrix.shape[0] - n_untrained
    if n_untrained == 0:
        return
    
    print(f'tokens with zero embeddings: {actual_bad_tokens[:10]} etc ...')

    where_untrained = where_untrained.tolist()
    actual_bad_tokens = tokenizer.convert_ids_to_tokens(where_untrained)
    actual_bad_tokens = [x for x in actual_bad_tokens if x is not None]

    sum_embedding = torch.sum(embedding_matrix, dtype=torch.float32, axis=0)
    sum_lm_head = torch.sum(lm_head_matrix, dtype=torch.float32, axis=0)

    sum_embedding -= torch.sum(
        embedding_matrix[where_untrained], dtype=torch.float32, axis=0
    )
    sum_lm_head -= torch.sum(
        lm_head_matrix[where_untrained], dtype=torch.float32, axis=0
    )

    mean_embedding = sum_embedding / n_trained
    mean_lm_head = sum_lm_head / n_trained

    mean_embedding = mean_embedding.repeat((n_untrained, 1))
    mean_lm_head = mean_lm_head.repeat((n_untrained, 1))

    embedding_matrix[where_untrained] = mean_embedding.to(embedding_matrix.dtype)
    lm_head_matrix[where_untrained] = mean_lm_head.to(lm_head_matrix.dtype)

    torch.cuda.empty_cache()