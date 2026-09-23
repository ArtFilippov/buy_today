"""Refine latent rank using validation only, then freeze the selection."""

import json
from time import perf_counter

import pandas as pd
from threadpoolctl import threadpool_limits

from buy_today.generation.histories import generate_histories
from buy_today.ranking.evaluation import evaluate_ranker
from buy_today.ranking.svd import SVDRanker
from buy_today.schema import read_dataset
from candidates import ProductDistance, FrequencyRanker, dataset_quality
from prepare_data import HERE, ROOT, ranking_data


def main():
    batch = read_dataset(ROOT / "data/olist-stream/batches/batch_000.csv")
    catalog = pd.DataFrame({"product_id": sorted(batch.product_id.unique())}, dtype="string")
    records = pd.read_csv(HERE / "refinement.csv").to_dict("records") if (HERE / "refinement.csv").exists() else []
    qualities = pd.read_csv(HERE / "refinement_quality.csv").to_dict("records") if (HERE / "refinement_quality.csv").exists() else []
    for seed in (42, 137, 2026):
        for temperature in (.1, .15, .2):
            name = f"category_price_t{temperature}"
            models = {f"svd_{n}": SVDRanker(n_components=n) for n in (128, 256, 384)}
            models["personal_frequency"] = FrequencyRanker(personal=True)
            if all(any(r["seed"] == seed and r["distance"] == name and r["model"] == model for r in records) for model in models):
                continue
            generated = generate_histories(
                batch, distance=ProductDistance(), temperature=temperature,
                batch_index=0, n_users=1000, random_state=seed,
            )
            qualities.append({"distance": name, "seed": seed, "temperature": temperature,
                              **dataset_quality(batch, generated.events, generated.anchors)})
            train = ranking_data(generated.events, catalog, "train")
            validation = ranking_data(generated.events, catalog, "validation")
            for model_name, model in models.items():
                started = perf_counter()
                model.fit(train)
                seconds = perf_counter() - started
                result = evaluate_ranker(model, validation, k=10)
                row = {"distance": name, "seed": seed, "model": model_name,
                       "fit_seconds": seconds, **result.metrics}
                records.append(row)
                print(json.dumps(row), flush=True)
            pd.DataFrame(records).to_csv(HERE / "refinement.csv", index=False)
            pd.DataFrame(qualities).to_csv(HERE / "refinement_quality.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
