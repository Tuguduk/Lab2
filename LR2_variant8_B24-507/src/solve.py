from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import linprog
import scipy

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
for folder in (RESULTS, FIGURES, LOGS):
    folder.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOGS / "solver_log.txt", level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", filemode="w"
)
logger = logging.getLogger("transport_lab")

def load_data():
    sup_df = pd.read_csv(DATA / "supply.csv")
    dem_df = pd.read_csv(DATA / "demand.csv")
    cost_df = pd.read_csv(DATA / "transport_cost.csv", index_col=0)
    cap_df = pd.read_csv(DATA / "route_capacity.csv", index_col=0)
    warehouses = sup_df["warehouse"].tolist()
    stores = dem_df["store"].tolist()
    supply = sup_df["supply_units"].to_numpy(dtype=float)
    demand = dem_df["demand_units"].to_numpy(dtype=float)
    cost = cost_df.loc[warehouses, stores].to_numpy(dtype=float)
    capacity = cap_df.loc[warehouses, stores].to_numpy(dtype=float)
    if cost.shape != (4, 6) or capacity.shape != (4, 6):
        raise ValueError("Для варианта 8 ожидаются матрицы стоимости и ёмкости 4x6.")
    if not np.isclose(supply.sum(), demand.sum()):
        raise ValueError("Базовая транспортная задача должна быть сбалансирована.")
    return warehouses, stores, supply, demand, cost, capacity

def build_matrices(supply, demand):
    n, m = len(supply), len(demand)
    Aeq, beq = [], []
    for i in range(n):
        row = np.zeros(n*m); row[i*m:(i+1)*m] = 1
        Aeq.append(row); beq.append(supply[i])
    for j in range(m):
        row = np.zeros(n*m)
        for i in range(n): row[i*m+j] = 1
        Aeq.append(row); beq.append(demand[j])
    return np.asarray(Aeq), np.asarray(beq)

def solve_model(supply, demand, cost, capacity):
    Aeq, beq = build_matrices(supply, demand)
    bounds = [(0, float(capacity[i,j])) for i in range(len(supply)) for j in range(len(demand))]
    return linprog(cost.ravel(), A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs-ds",
                   options={"presolve": True})

def independent_check(flow, supply, demand, cost, capacity, objective):
    # Проверка выполняется только средствами NumPy, без обращения к объекту HiGHS.
    warehouse_error = flow.sum(axis=1) - supply
    store_error = flow.sum(axis=0) - demand
    cap_violation = np.maximum(flow - capacity, 0)
    nonneg_violation = np.maximum(-flow, 0)
    recalculated = float(np.sum(flow * cost))
    checks = [
        ("warehouse_balance_max_abs", float(np.max(np.abs(warehouse_error)))),
        ("store_balance_max_abs", float(np.max(np.abs(store_error)))),
        ("capacity_violation_max", float(np.max(cap_violation))),
        ("nonnegativity_violation_max", float(np.max(nonneg_violation))),
        ("objective_difference", abs(recalculated - float(objective)))
    ]
    df = pd.DataFrame([{"check": k, "value": v, "tolerance": 1e-7, "passed": v <= 1e-7} for k,v in checks])
    df.to_csv(RESULTS / "independent_check.csv", index=False)
    return df

def greedy_heuristic(supply, demand, cost, capacity):
    # Жадная эвристика: магазины обслуживаются в порядке наименьшего отношения
    # суммарной доступной ёмкости к спросу; каждый магазин получает груз по
    # самым дешёвым доступным маршрутам.
    order = np.argsort(capacity.sum(axis=0) / demand)
    remaining_supply = supply.astype(int).copy()
    remaining_demand = demand.astype(int).copy()
    flow = np.zeros_like(capacity, dtype=float)
    for j in order:
        for i in np.argsort(cost[:,j]):
            q = min(remaining_supply[i], remaining_demand[j], int(capacity[i,j]))
            if q > 0:
                flow[i,j] += q
                remaining_supply[i] -= q
                remaining_demand[j] -= q
            if remaining_demand[j] == 0:
                break
    feasible = bool(np.all(remaining_supply == 0) and np.all(remaining_demand == 0))
    return flow, feasible, float(np.sum(flow*cost)) if feasible else None, order

def draw_network(warehouses, stores, flow, capacity, cost):
    G = nx.DiGraph()
    for w in warehouses: G.add_node(w, bipartite=0)
    for s in stores: G.add_node(s, bipartite=1)
    for i,w in enumerate(warehouses):
        for j,s in enumerate(stores):
            G.add_edge(w,s,flow=flow[i,j],cap=capacity[i,j],cost=cost[i,j])
    pos = {w:(0,3-i) for i,w in enumerate(warehouses)}
    pos.update({s:(3,5-j) for j,s in enumerate(stores)})
    plt.figure(figsize=(12,7))
    nx.draw_networkx_nodes(G,pos,nodelist=warehouses,node_color="#4C78A8",node_size=1600)
    nx.draw_networkx_nodes(G,pos,nodelist=stores,node_color="#F2CF5B",node_size=1500)
    nx.draw_networkx_labels(G,pos,font_size=10,font_weight="bold")
    labels={(u,v):f"{d['flow']:.0f}/{d['cap']} | c={d['cost']}" for u,v,d in G.edges(data=True) if d["flow"]>1e-7}
    nx.draw_networkx_edges(G,pos,alpha=0.25,arrows=True,arrowsize=12,width=1.2)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,font_size=7,label_pos=0.5)
    plt.title("Оптимальные перевозки: поток / пропускная способность; тариф")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(FIGURES/"network_optimal.png",dpi=180); plt.close()

def run():
    warehouses, stores, supply, demand, cost, capacity = load_data()
    result = solve_model(supply,demand,cost,capacity)
    logger.info("SciPy version: %s", scipy.__version__)
    logger.info("Method: highs-ds; presolve=True")
    logger.info("Base status: %s; message: %s", result.status, result.message)
    if not result.success:
        raise RuntimeError(result.message)
    flow = result.x.reshape(len(warehouses),len(stores))
    pd.DataFrame(flow,index=warehouses,columns=stores).rename_axis("warehouse").to_csv(RESULTS/"optimal_flows.csv")
    rows=[]
    for i,w in enumerate(warehouses):
        for j,s in enumerate(stores):
            rows.append({"warehouse":w,"store":s,"flow":flow[i,j],"capacity":capacity[i,j],
                         "cost_per_unit":cost[i,j],"utilization":flow[i,j]/capacity[i,j]})
    pd.DataFrame(rows).to_csv(RESULTS/"flows_detailed.csv",index=False)

    checks=independent_check(flow,supply,demand,cost,capacity,result.fun)
    if not checks["passed"].all():
        raise AssertionError("Независимая проверка не пройдена.")

    utilization = flow / capacity
    i,j = np.unravel_index(np.argmax(utilization),utilization.shape)
    failed_capacity = capacity.copy(); failed_capacity[i,j] = 0
    failed = solve_model(supply,demand,cost,failed_capacity)
    increased_demand = np.ceil(demand*1.2)
    growth = solve_model(supply,increased_demand,cost,capacity)
    logger.info("Failure arc %s-%s status=%s objective=%s",warehouses[i],stores[j],failed.status,failed.fun)
    logger.info("Demand +20%% status=%s message=%s",growth.status,growth.message)

    scenarios=[
      {"scenario":"Базовый","status":"optimal","objective":result.fun,"delta_vs_base":0.0},
      {"scenario":f"Отказ дуги {warehouses[i]}-{stores[j]}",
       "status":"optimal" if failed.success else "infeasible",
       "objective":failed.fun if failed.success else np.nan,
       "delta_vs_base":failed.fun-result.fun if failed.success else np.nan},
      {"scenario":"Рост спроса на 20%","status":"optimal" if growth.success else "infeasible",
       "objective":growth.fun if growth.success else np.nan,"delta_vs_base":np.nan}
    ]
    pd.DataFrame(scenarios).to_csv(RESULTS/"scenario_results.csv",index=False)

    dual=[]
    for k,w in enumerate(warehouses): dual.append({"constraint":f"supply_{w}","marginal":result.eqlin.marginals[k]})
    for k,s in enumerate(stores): dual.append({"constraint":f"demand_{s}","marginal":result.eqlin.marginals[len(warehouses)+k]})
    for k,w in enumerate(warehouses):
        for l,s in enumerate(stores):
            marginal=result.upper.marginals[k*len(stores)+l]
            if abs(marginal)>1e-9: dual.append({"constraint":f"route_capacity_{w}_{s}","marginal":marginal})
    pd.DataFrame(dual).to_csv(RESULTS/"dual_prices.csv",index=False)

    gflow,gfeasible,gcost,gorder=greedy_heuristic(supply,demand,cost,capacity)
    pd.DataFrame(gflow,index=warehouses,columns=stores).rename_axis("warehouse").to_csv(RESULTS/"greedy_flows.csv")
    greedy_info={"feasible":gfeasible,"objective":gcost,"gap_percent":(gcost/result.fun-1)*100 if gfeasible else None}
    (RESULTS/"greedy_summary.json").write_text(json.dumps(greedy_info,indent=2),encoding="utf-8")
    draw_network(warehouses,stores,flow,capacity,cost)
    plt.figure(figsize=(9,4.8)); plt.imshow(flow,aspect="auto"); plt.colorbar(label="Объём перевозки")
    plt.xticks(range(len(stores)),stores); plt.yticks(range(len(warehouses)),warehouses)
    plt.xlabel("Магазины"); plt.ylabel("Склады"); plt.title("Матрица оптимальных потоков")
    plt.tight_layout(); plt.savefig(FIGURES/"flow_heatmap.png",dpi=180); plt.close()
    if gfeasible:
        plt.figure(figsize=(7,4.5)); plt.bar(["HiGHS optimum","Greedy heuristic"],[result.fun,gcost])
        plt.ylabel("Общая стоимость"); plt.title("Сравнение оптимального плана и жадной эвристики")
        plt.tight_layout(); plt.savefig(FIGURES/"greedy_comparison.png",dpi=180); plt.close()

    summary={
      "status":result.message,"objective":float(result.fun),"max_independent_violation":float(checks["value"].max()),
      "failure_arc":f"{warehouses[i]}-{stores[j]}","failure_status":failed.message,
      "failure_objective":float(failed.fun) if failed.success else None,
      "demand_plus_20_status":growth.message,"greedy":greedy_info
    }
    (RESULTS/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    logger.info("Objective: %.6f",result.fun)
    logger.info("Independent checks: %s",checks["passed"].all())
    logger.info("Greedy: %s",greedy_info)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    run()
