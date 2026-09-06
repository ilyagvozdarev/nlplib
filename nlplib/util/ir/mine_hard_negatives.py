import argparse, random, logging
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import sentence_transformers.util as st

from hybrid_retrieval.util.io import read_config, read_json

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


MINE_CONFIG_DEFAULT = dict(
    semantic=dict(
        range_min=5,
        range_max=35,
        num_negatives=30,
        n_top=1,
        n_rand=1
    ),
    bm_25=dict(
        top_k=8
    )
)

ARGS_DEFAULT = dict(
    model              = dict(default="deepvk/USER-bge-m3"),
    mine_config        = dict(default="configs/mine_config.yaml"),
    out_file           = dict(default="output.json"),
    qrels              = dict(default="qrels.json"),
    passages           = dict(default="passages.json"),
    queries            = dict(default="queries.json"),
    cache_embeddings   = dict(default="cache_embeddings"),
    batch_size         = dict(default=64),
    seed               = dict(default=42)
)


def mine_bm25(corpus, queries_text, passage_to_id, mine_config):
    import string
    import numpy as np
    from rank_bm25 import BM25Okapi
    from hybrid_retrieval.util.stop_words import remove_stop_words

    def bm25_process(text):
        tokenized_doc = [token.strip(string.punctuation) for token in text.lower().split()]
        tokenized_doc = [token for token in tokenized_doc if len(token) > 0]
        tokenized_doc = remove_stop_words(tokenized_doc)
        return tokenized_doc

    def search_bm25(query, bm25, top_k=8):
        bm25_scores = bm25.get_scores(bm25_process(query))
        top_n = np.argpartition(bm25_scores, -top_k-2)[-top_k-2:]
        bm25_hits = [{"corpus_id": idx, "score": bm25_scores[idx]} for idx in top_n]
        bm25_hits = sorted(bm25_hits, key=lambda x: x["score"], reverse=True)
        return bm25_hits[0:top_k]

    tokenized_corpus = []
    for passage in corpus:
        tokenized_corpus.append(bm25_process(passage))

    bm25 = BM25Okapi(tokenized_corpus)      # default k1=1.5, b=0.75, epsilon=0.25

    bm25_mined = [search_bm25(q, bm25, top_k=mine_config["top_k"]) for q in queries_text]

    bm25_mined_ids = [
        [passage_to_id[corpus[p['corpus_id']]] for p in mined] 
        for mined in bm25_mined
    ]
    logging.info(f"bm25_mined_ids[0]: {bm25_mined_ids[0]}")

    return bm25_mined_ids


def mine_hard_negatives(
    model,
    qrels_dataset,
    passages,
    queries,
    mine_config = MINE_CONFIG_DEFAULT,
    batch_size = ARGS_DEFAULT["batch_size"],
    seed = ARGS_DEFAULT["seed"],
    out_file = None,
    cache_embeddings = None
):
    query_to_id = {text:qid for qid, text in queries.items()}
    passage_to_id = {text:pid for pid, text in passages.items()}

    if isinstance(model, str): 
        model = SentenceTransformer(
            model, 
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        model.eval()

    logging.info(f"qrels_dataset: {qrels_dataset}")

    if cache_embeddings:
        cache_embeddings = Path(cache_embeddings)
        cache_embeddings.mkdir(parents=True, exist_ok=True)

    corpus = list(passages.values())
    args_ = dict(
        dataset=qrels_dataset,
        corpus=corpus,
        cache_folder=cache_embeddings,
        use_faiss=True,
        batch_size=batch_size   
    )

    mine_args = mine_config["semantic"]
    hard_negatives = st.mine_hard_negatives(
        **args_,
        model=model,
        range_min=mine_args["range_min"],
        range_max=mine_args["range_max"],
        num_negatives=mine_args["num_negatives"],
        sampling_strategy="top",
        output_format="labeled-list", 
        # output_scores=True
    )

    logging.info(f"hard_negatives {hard_negatives}")

    N_TOP, N_RAND = mine_args["n_top"], mine_args["n_rand"]
    ANCHOR, POS = "anchor", "positive"
    rng = random.Random(seed)

    discarded = 0
    def sample_negatives(batch):
        nonlocal discarded
        out = {ANCHOR: [], POS: [], **{f"negative_{i+1}": [] for i in range(N_TOP + N_RAND)}}
        for q, docs in zip(batch[ANCHOR], batch[POS]):
            negs = docs[1:]
            if len(negs) < N_TOP + N_RAND:
                discarded += 1
                continue                      # недобрали — выбрасываем строку
            chosen = negs[:N_TOP] + rng.sample(negs[N_TOP:], N_RAND)
            out[ANCHOR].append(q)
            out[POS].append(docs[0])
            for i, n in enumerate(chosen):
                out[f"negative_{i+1}"].append(n)
        return out

    ds_train = hard_negatives.map(
        sample_negatives, 
        batched=True, 
        remove_columns=hard_negatives.column_names
    )

    if discarded:
        logging.info(f'!!! len(negs) < N_TOP + N_RAND: {discarded}')

    bm25_mined_ids = None
    if 'bm_25' in mine_config:
        id_cols = {
            col: [(query_to_id if col == "anchor" else passage_to_id)[text] for text in ds_train[col]]
            for col in ds_train.column_names
        }
        ds_train_ids = [dict(zip(id_cols, vals)) for vals in zip(*id_cols.values())]
        logging.info(f"ds_train_ids: {ds_train_ids[:2]}")

        assert len(passage_to_id) == len(passages), "дубли текстов пассажей ломают id-маппинг"
        assert len(query_to_id) == len(queries)
    
        queries_text = list(ds_train['anchor'])

        bm25_mined_ids = mine_bm25(corpus, queries_text, passage_to_id, mine_config["bm_25"])

    if bm25_mined_ids is not None:
        import pandas as pd
        # merge semantic + bm25 hards
        discarded_counts = []
        bm25_hards = []
        for ps, bm25_ids in zip(ds_train_ids, bm25_mined_ids):
            ps = [v for k, v in ps.items() if k != 'anchor']
            bm25_ids_ = [p for p in bm25_ids if p not in ps]
            discarded_counts.append(len(bm25_ids) - len(bm25_ids_))
            bm25_hard_1 = rng.sample(bm25_ids_[:2], 1)[0]
            bm25_hards.append(bm25_hard_1)
        logging.info(f'discarded count: {pd.DataFrame(discarded_counts).value_counts().sort_index()}')

        ds_train = ds_train.add_column(f'negative_{len(ds_train.column_names)-1}', [passages[pid] for pid in bm25_hards])
        logging.info(f"ds_train: {ds_train}")

    if out_file:
        ds_train.to_json(out_file)

    return ds_train


def read_data(args):
    args = vars(args)
    fields = ['mine_config', 'qrels', 'queries', 'passages']
    for field in fields:
        args[field] = (read_config if field == 'mine_config' else read_json)(args[field])

    def to_text(batch):
        return {
            'anchor': [args["queries"][id_] for id_ in batch['qid']],
            'positive': [args["passages"][id_] for id_ in batch['pid']]
        }

    qrels_dataset = Dataset.from_list(
        [{'qid': qid, 'pid': pid} for qid, pid in args["qrels"].items()]
    )
    qrels_dataset = qrels_dataset.map(
        to_text, batched=True, batch_size=1000, remove_columns=["qid", "pid"]
    )

    args["qrels_dataset"] = qrels_dataset
    del args["qrels"]

    return args


def parse_args(args_default):
    parser = argparse.ArgumentParser()
    for name, v in args_default.items():
        parser.add_argument(f"--{name}", **{"type": type(v["default"]), **v})
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args(ARGS_DEFAULT)
    args = read_data(args)
    mine_hard_negatives(**args)