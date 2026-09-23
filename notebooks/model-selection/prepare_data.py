"""Run validation-only screening; final evaluation is a separate explicit stage."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from buy_today.clustering.distances.temporal_distances import TimestampDistance
from buy_today.generation.histories import generate_histories
from buy_today.ranking.data import RankingData
from buy_today.ranking.evaluation import evaluate_ranker
from buy_today.ranking.random import RandomRanker
from buy_today.ranking.svd import SVDRanker
from buy_today.schema import read_dataset
from candidates import ProductDistance, FrequencyRanker, UserKNNRanker, dataset_quality

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def configurations():
    result = {
        f"timestamp_{days:g}d": (TimestampDistance(), 86400 * days)
        for days in (.125, .5, 1, 3)
    }
    result["uniform"] = (ProductDistance(category_weight=0, price_weight=0), 1.0)
    result["category"] = (ProductDistance(price_weight=0), .15)
    for temperature in (.1, .2, .4):
        result[f"category_price_t{temperature}"] = (ProductDistance(), temperature)
    for temperature in (.2, .4):
        result[f"category_price_time_t{temperature}"] = (
            ProductDistance(time_scale_days=30), temperature,
        )
    return result


def rankers():
    return {
        "random": RandomRanker(random_state=42),
        "popularity": FrequencyRanker(),
        "personal_frequency": FrequencyRanker(personal=True),
        **{f"svd_{n}": SVDRanker(n_components=n, random_state=42) for n in (16, 32, 64, 128)},
        **{f"user_knn_{n}": UserKNNRanker(n_neighbors=n) for n in (20, 50)},
    }


def ranking_data(events, catalog, split):
    return RankingData(events.loc[events.split == split, ["user_id", "product_id"]], catalog)


def screen(users, seeds):
    source = ROOT / "data/olist-stream/batches/batch_000.csv"
    manifest = {
        "source": str(source.relative_to(ROOT)),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "users": users, "seeds": seeds, "split_sizes": [70, 15, 15], "k": 10,
        "selection_split": "validation", "test_used_for_selection": False,
        "quality_gates": {"unique_train_median_min": 30, "unique_train_p10_min": 10,
                          "mean_top_product_share_max": .2, "category_tvd_max": .1},
    }
    manifest_path = HERE / "screening_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Screening inputs changed; use a separate result directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    batch = read_dataset(source)
    catalog = pd.DataFrame({"product_id": sorted(batch.product_id.unique())}, dtype="string")
    records = pd.read_csv(HERE / "validation.csv").to_dict("records") if (HERE / "validation.csv").exists() else []
    quality = pd.read_csv(HERE / "quality.csv").to_dict("records") if (HERE / "quality.csv").exists() else []
    for seed in seeds:
        for name, (distance, temperature) in configurations().items():
            if any(row["distance"] == name and row["seed"] == seed for row in quality):
                continue
            started = perf_counter()
            generated = generate_histories(
                batch, distance=distance, batch_index=0, n_users=users,
                temperature=temperature, random_state=seed,
            )
            quality.append({
                "distance": name, "seed": seed, "temperature": temperature,
                **dataset_quality(batch, generated.events, generated.anchors),
                "generation_seconds": perf_counter() - started,
            })
            train = ranking_data(generated.events, catalog, "train")
            validation = ranking_data(generated.events, catalog, "validation")
            for model_name, model in rankers().items():
                started = perf_counter()
                model.fit(train)
                fit_seconds = perf_counter() - started
                evaluation = evaluate_ranker(model, validation, k=10)
                row = {
                    "distance": name, "seed": seed, "model": model_name,
                    "fit_seconds": fit_seconds, **evaluation.metrics,
                }
                records.append(row)
                print(json.dumps(row), flush=True)
            pd.DataFrame(records).to_csv(HERE / "validation.csv", index=False)
            pd.DataFrame(quality).to_csv(HERE / "quality.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2026])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        screen(args.users, args.seeds)
