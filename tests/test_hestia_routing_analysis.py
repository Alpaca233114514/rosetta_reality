import numpy as np

from scripts.analyze_hestia_routing_effects import classify


def entry(values):
    error = {k: dict(mean=float(np.mean(values)), by_noise=values) for k in ("mae", "mse")}
    return dict(standard=dict(dev5=dict(full=dict(left_joint=dict(errors=dict(model=error))))))


def test_routing_classification_does_not_force_a_unique_winner():
    metrics = {
        "base640_native": entry([10] * 4),
        "base640_rfull": entry([5] * 4),
        "base640_rintra": entry([6] * 4),
        "base640_rmass": entry([9] * 4),
    }
    assert classify(metrics)["classification"] == "rintra"
    metrics["base640_rmass"] = entry([6] * 4)
    assert classify(metrics)["classification"] == "both_components_individually_qualify"
    metrics["base640_rintra"] = entry([8] * 4)
    metrics["base640_rmass"] = entry([8] * 4)
    assert classify(metrics)["classification"] == "mixed_or_interaction"


def test_one_adverse_noise_prevents_dominance_despite_good_mean():
    metrics = {
        "base640_native": entry([10] * 4),
        "base640_rfull": entry([5] * 4),
        "base640_rintra": entry([11, 4, 4, 4]),
        "base640_rmass": entry([9] * 4),
    }
    result = classify(metrics)
    assert result["components"][0]["fraction_of_full_pair_mse_benefit"] > 0.8
    assert result["classification"] == "mixed_or_interaction"
