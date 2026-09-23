"""Evaluate frozen parameters on new synthetic seeds; publish usable snapshots."""

import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from buy_today.clustering.distances import CategoryPriceDistance, TimestampDistance
from buy_today.generation.histories import generate_histories
from buy_today.generation.storage import generate_dataset, read_history_dataset
from buy_today.ranking.data import read_ranking_data
from buy_today.ranking.evaluation import evaluate_ranker
from buy_today.ranking.random import RandomRanker
from buy_today.ranking.storage import train_ranker, report_ranking
from buy_today.ranking.svd import SVDRanker
from buy_today.schema import read_dataset
from candidates import FrequencyRanker, ProductDistance, dataset_quality
from prepare_data import HERE, ROOT, ranking_data


def novelty(model, train, held_out, k):
    seen = train.interactions.groupby("user_id").product_id.agg(set)
    rows = []
    for user, group in held_out.interactions.groupby("user_id", sort=True):
        relevant = set(group.product_id)
        novel = relevant - seen[user]
        recommendations = set(model.predict(user, k))
        rows.append({"user_id": user, "n_relevant": len(relevant),
                     "n_novel_relevant": len(novel),
                     "novel_hits": len(novel & recommendations),
                     "seen_hits": len((relevant & seen[user]) & recommendations),
                     "novel_recall": len(novel & recommendations) / len(novel) if novel else np.nan})
    return pd.DataFrame(rows)


def main():
    selection = json.loads((HERE / "selection.json").read_text())
    batch_path = ROOT / "data/olist-stream/batches/batch_000.csv"
    batch = read_dataset(batch_path)
    model_root = ROOT / "models/model-selection"
    model_root.mkdir(parents=True, exist_ok=True)
    distance_path = model_root / "distance.joblib"
    distance = CategoryPriceDistance().fit(batch)
    # Production implementation must exactly match the candidate used for selection.
    probe = batch.sample(50, random_state=73)
    np.testing.assert_allclose(distance.pairwise(probe), ProductDistance().pairwise(probe))
    joblib.dump(distance, distance_path)
    rows, per_user, qualities, novel_rows = [], [], [], []
    for seed in selection["final_seeds"]:
        dataset = ROOT / f"data/model-selection/seed_{seed}"
        if not dataset.exists():
            generate_dataset(batch_path, distance_path, dataset,
                             temperature=selection["temperature"],
                             n_users=selection["final_users"], random_state=seed)
        generated = read_history_dataset(dataset)
        qualities.append({"seed": seed, **dataset_quality(batch, generated.events, generated.anchors)})
        train = read_ranking_data(dataset, split="train")
        test = read_ranking_data(dataset, split="test")
        validation = read_ranking_data(dataset, split="validation")
        directory = model_root / f"seed_{seed}" / "ranking"
        if not directory.exists():
            train_ranker(dataset, directory, ranker=SVDRanker(**selection["ranker_parameters"]))
        for split in ("validation", "test"):
            if not (directory / split).exists():
                report_ranking(dataset, directory, directory / split, split=split, k=selection["k"])
        models = {"svd_256": joblib.load(directory / "model.joblib"),
                  "svd_32": SVDRanker(n_components=32).fit(train),
                  "personal_frequency": FrequencyRanker(personal=True).fit(train),
                  "popularity": FrequencyRanker().fit(train),
                  "random": RandomRanker().fit(train)}
        for name, model in models.items():
            for split, held_out in (("validation", validation), ("test", test)):
                evaluation = evaluate_ranker(model, held_out, k=selection["k"])
                row = {"seed": seed, "distance": "category_price_t0.1", "model": name,
                       "split": split, **evaluation.metrics}
                rows.append(row)
                print(json.dumps(row), flush=True)
                if split == "test":
                    per_user.append(evaluation.per_user.assign(seed=seed, model=name))
            if name in ("svd_256", "personal_frequency"):
                novel_rows.append(novelty(model, train, test, selection["k"]).assign(seed=seed, model=name))
        # Compare against the current pipeline's default task; label as a different dataset.
        original = generate_histories(batch, distance=TimestampDistance(), batch_index=0,
                                      n_users=selection["final_users"], temperature=86400., random_state=seed)
        original_train = ranking_data(original.events, train.catalog, "train")
        original_test = ranking_data(original.events, train.catalog, "test")
        evaluation = evaluate_ranker(SVDRanker().fit(original_train), original_test, k=selection["k"])
        rows.append({"seed": seed, "distance": "timestamp_1d", "model": "svd_32",
                     "split": "test", **evaluation.metrics})
        pd.DataFrame(rows).to_csv(HERE / "final_metrics.csv", index=False)
        pd.concat(per_user, ignore_index=True).to_csv(HERE / "test_per_user.csv", index=False)
        pd.DataFrame(qualities).to_csv(HERE / "final_quality.csv", index=False)
        pd.concat(novel_rows, ignore_index=True).to_csv(HERE / "novelty.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
