from pathlib import Path
import json
import numpy as np
import pandas as pd

SEED = 10013
WAREHOUSES = [f"W{i}" for i in range(1, 5)]
STORES = [f"S{j}" for j in range(1, 7)]

def generate(output_dir="data"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    demand = rng.integers(38, 61, size=6).astype(int)
    total = int(demand.sum())
    raw = rng.random(4)
    supply_float = raw / raw.sum() * total
    supply = np.floor(supply_float).astype(int)
    for k in np.argsort(supply_float - supply)[::-1][:total - int(supply.sum())]:
        supply[k] += 1

    cost = rng.integers(4, 13, size=(4, 6)).astype(int)
    capacity = np.zeros((4, 6), dtype=int)
    for i in range(4):
        for j in range(6):
            base = min(supply[i], demand[j])
            capacity[i, j] = max(12, int(np.floor(base * rng.uniform(0.52, 0.88))))

    # Guarantee enough aggregate outgoing and incoming route capacity.
    for i in range(4):
        while capacity[i].sum() < supply[i]:
            j = int(np.argmax(np.minimum(supply[i], demand) - capacity[i]))
            capacity[i, j] += min(5, max(1, supply[i] - capacity[i].sum()))
    for j in range(6):
        while capacity[:, j].sum() < demand[j]:
            i = int(np.argmax(np.minimum(supply, demand[j]) - capacity[:, j]))
            capacity[i, j] += min(5, max(1, demand[j] - capacity[:, j].sum()))

    pd.DataFrame({"warehouse": WAREHOUSES, "supply_units": supply}).to_csv(out / "supply.csv", index=False)
    pd.DataFrame({"store": STORES, "demand_units": demand}).to_csv(out / "demand.csv", index=False)
    pd.DataFrame(cost, index=WAREHOUSES, columns=STORES).rename_axis("warehouse").to_csv(out / "transport_cost.csv")
    pd.DataFrame(capacity, index=WAREHOUSES, columns=STORES).rename_axis("warehouse").to_csv(out / "route_capacity.csv")

    meta = {
        "variant": 8, "group": "Б24-507", "N": 13, "Q": 0, "Y": 26,
        "seed": SEED, "warehouses": 4, "stores": 6,
        "flow_unit": "условная единица груза",
        "cost_unit": "условная денежная единица за единицу груза",
        "data_type": "synthetic",
        "generation_note": "Данные не предоставлены преподавателем. Синтетические значения сгенерированы NumPy Generator (PCG64), seed=10013."
    }
    (out / "instance.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return supply, demand, cost, capacity

if __name__ == "__main__":
    generate()
