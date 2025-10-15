import csv
import random
import math
import os
import numpy as np

GRID_X_MIN, GRID_X_MAX = 0, 600
GRID_Y_MIN, GRID_Y_MAX = 0, 600


def manhattan_distance(x1, y1, x2, y2):
    return abs(x2 - x1) + abs(y2 - y1)


PROFILES = {
    "easy": {
        "utilization": 0.65,
        "clusters": 2,
        "slack_ratio": 0.8,
        "cluster_spread": 80,
        "conflicting_windows_ratio": 0.2,
    },
    "medium": {
        "utilization": 0.75,
        "clusters": 3,
        "slack_ratio": 0.5,
        "cluster_spread": 60,
        "conflicting_windows_ratio": 0.4,
    },
    "hard": {
        "utilization": 0.85,
        "clusters": 4,
        "slack_ratio": 0.3,
        "cluster_spread": 50,
        "conflicting_windows_ratio": 0.6,
    },
    "extreme": {
        "utilization": 0.90,
        "clusters": 5,
        "slack_ratio": 0.15,
        "cluster_spread": 40,
        "conflicting_windows_ratio": 0.8,
    }
}


def generate_instance(num_scs, num_wis,
                      filename="instance.csv",
                      horizon=None,
                      difficulty="hard",
                      seed=42):


    random.seed(seed)
    np.random.seed(seed)

    cfg = PROFILES[difficulty]

    scs = []
    if difficulty in ["hard", "extreme"]:
        positions = [
            (50, 50), (550, 50), (50, 550), (550, 550),
            (300, 50), (50, 300), (550, 300), (300, 550)
        ]
        for i in range(num_scs):
            if i < len(positions):
                x, y = positions[i]
                x += random.uniform(-20, 20)
                y += random.uniform(-20, 20)
            else:
                x = random.uniform(GRID_X_MIN + 50, GRID_X_MAX - 50)
                y = random.uniform(GRID_Y_MIN + 50, GRID_Y_MAX - 50)

            sc = {
                "id": f"SC{i}",
                "x": x,
                "y": y,
                "start_time": 0
            }
            scs.append(sc)
    else:
        for i in range(num_scs):
            sc = {
                "id": f"SC{i}",
                "x": random.uniform(GRID_X_MIN + 50, GRID_X_MAX - 50),
                "y": random.uniform(GRID_Y_MIN + 50, GRID_Y_MAX - 50),
                "start_time": 0
            }
            scs.append(sc)

    # 2. Generate cluster centers
    clusters = []
    for _ in range(cfg["clusters"]):
        cluster_x = random.uniform(GRID_X_MIN + 100, GRID_X_MAX - 100)
        cluster_y = random.uniform(GRID_Y_MIN + 100, GRID_Y_MAX - 100)
        clusters.append((cluster_x, cluster_y))

    # 3. Generate WIs
    wis = []
    for j in range(num_wis):
        cluster_x, cluster_y = random.choice(clusters)
        spread = cfg["cluster_spread"]

        o_x = np.clip(random.gauss(cluster_x, spread), GRID_X_MIN, GRID_X_MAX)
        o_y = np.clip(random.gauss(cluster_y, spread), GRID_Y_MIN, GRID_Y_MAX)

        if difficulty in ["hard", "extreme"] and random.random() < 0.3:
            other_cluster = random.choice(
                [c for c in clusters if c != (cluster_x, cluster_y)] or [(cluster_x, cluster_y)])
            d_x = np.clip(random.gauss(other_cluster[0], spread), GRID_X_MIN, GRID_X_MAX)
            d_y = np.clip(random.gauss(other_cluster[1], spread), GRID_Y_MIN, GRID_Y_MAX)
        else:
            d_x = np.clip(random.gauss(cluster_x, spread), GRID_X_MIN, GRID_X_MAX)
            d_y = np.clip(random.gauss(cluster_y, spread), GRID_Y_MIN, GRID_Y_MAX)

        while manhattan_distance(o_x, o_y, d_x, d_y) < 30:
            d_x = np.clip(random.gauss(cluster_x, spread), GRID_X_MIN, GRID_X_MAX)
            d_y = np.clip(random.gauss(cluster_y, spread), GRID_Y_MIN, GRID_Y_MAX)

        op_time = int(manhattan_distance(o_x, o_y, d_x, d_y))
        op_time = max(20, op_time)

        wi = {
            "id": f"WI{j}",
            "o_x": o_x,
            "o_y": o_y,
            "d_x": d_x,
            "d_y": d_y,
            "op_time": op_time
        }
        wis.append(wi)

    wi_by_id = {w["id"]: w for w in wis}

    print(f"\nBuilding greedy schedule (urgent-first strategy)...")

    sc_state = {sc["id"]: {"x": sc["x"], "y": sc["y"], "time": sc["start_time"]} for sc in scs}


    sorted_wis = sorted(wis, key=lambda w: w["op_time"])

    schedule = {}
    routes = {sc["id"]: [] for sc in scs}

    for wi in sorted_wis:
        best_sc = None
        best_end = float('inf')
        best_start = None

        for sc in scs:
            state = sc_state[sc["id"]]
            x, y, current_time = state["x"], state["y"], state["time"]

            # Empty travel to pickup
            empty_travel = manhattan_distance(x, y, wi["o_x"], wi["o_y"])
            arrival_at_pickup = current_time + empty_travel

            # Pickup at arrival
            pickup_time = arrival_at_pickup

            # Dropoff after operation
            dropoff_time = pickup_time + wi["op_time"]

            # Selection: earliest dropoff
            if dropoff_time < best_end:
                best_end = dropoff_time
                best_start = pickup_time
                best_sc = sc["id"]

        # Assign to best SC
        schedule[wi["id"]] = (best_start, best_end, best_sc)
        routes[best_sc].append(wi["id"])

        # Update SC state
        sc_state[best_sc]["x"] = wi["d_x"]
        sc_state[best_sc]["y"] = wi["d_y"]
        sc_state[best_sc]["time"] = best_end

    # Sort routes by start time
    for scid in routes:
        routes[scid] = sorted(routes[scid], key=lambda wid: schedule[wid][0])

    # 5. Set horizon
    max_end_time = max(end for _, end, _ in schedule.values())
    if horizon is None:
        horizon = int(max_end_time * 1.15)

    # 6. Create SEPARATE time windows for pickup and dropoff - MORE GENEROUS
    slack_ratio = cfg["slack_ratio"]
    print(f"Creating time windows (slack_ratio={slack_ratio})...")

    for wi in wis:
        scheduled_pickup_time, scheduled_dropoff_time, _ = schedule[wi["id"]]
        op_time = wi["op_time"]

        # Calculate slack for pickup window - MORE GENEROUS FOR LARGE INSTANCES
        base_slack = int(op_time * slack_ratio)
        if num_wis > 80:
            base_slack = int(base_slack * 1.5)
        pickup_slack = max(10, base_slack)

        # PICKUP WINDOW [ES, LS]
        ES = max(0, int(scheduled_pickup_time - pickup_slack))
        LS = min(horizon - op_time, int(scheduled_pickup_time + pickup_slack))

        if LS < ES:
            LS = ES

        # Calculate slack for dropoff window
        dropoff_slack = pickup_slack
        dropoff_slack = max(10, dropoff_slack)

        # DROPOFF WINDOW [EE, LE]
        EE = max(0, int(scheduled_dropoff_time - dropoff_slack))
        LE = min(horizon, int(scheduled_dropoff_time + dropoff_slack))

        if LE < EE:
            LE = EE

        # CRITICAL CONSTRAINT: LS + op_time <= LE
        if LS + op_time > LE:
            LE = min(horizon, LS + op_time)

        wi["ES"] = ES
        wi["LS"] = LS
        wi["EE"] = EE
        wi["LE"] = LE

    # 7. Create CONFLICTING windows (bottlenecks) - LESS AGGRESSIVE
    num_conflicts = int(num_wis * cfg["conflicting_windows_ratio"])

    # Reduce bottlenecks for large instances
    if num_wis > 80:
        num_conflicts = int(num_conflicts * 0.7)

    if num_conflicts > 0 and difficulty != "easy":
        num_bottlenecks = min(3, max(2, cfg["clusters"]))
        period_width = horizon // (num_bottlenecks + 1)

        bottleneck_periods = []
        for i in range(num_bottlenecks):
            center = (i + 1) * horizon // (num_bottlenecks + 1)
            bottleneck_periods.append((center - period_width // 4, center + period_width // 4))

        wis_to_bottleneck = random.sample(wis, min(num_conflicts, len(wis)))

        for wi in wis_to_bottleneck:
            period_start, period_end = random.choice(bottleneck_periods)
            op_time = wi["op_time"]

            # Tighter slack for bottleneck - but not too tight
            bottleneck_slack = int(op_time * slack_ratio * 0.6)
            bottleneck_slack = max(5, bottleneck_slack)

            # Force pickup into bottleneck period
            ES = max(0, period_start)
            LS = min(horizon - op_time, period_end)

            if LS < ES:
                LS = min(horizon - op_time, ES + 5)

            # Dropoff window
            EE = max(0, period_start + op_time)
            LE = min(horizon, period_end + op_time + bottleneck_slack)

            if LE < EE:
                LE = EE

            # Ensure feasibility: LS + op_time <= LE
            if LS + op_time > LE:
                LE = min(horizon, LS + op_time)

            # Validate before assignment
            if ES <= LS and EE <= LE and LE <= horizon and LS + op_time <= LE:
                wi["ES"] = ES
                wi["LS"] = LS
                wi["EE"] = EE
                wi["LE"] = LE

    # 8. Verify consistency
    print("Validating time window consistency...")
    violations = []

    for wi in wis:
        if wi["ES"] > wi["LS"]:
            violations.append(f"{wi['id']}: ES > LS")
        if wi["EE"] > wi["LE"]:
            violations.append(f"{wi['id']}: EE > LE")
        if wi["LE"] > horizon:
            violations.append(f"{wi['id']}: LE > horizon")
        if wi["LS"] + wi["op_time"] > wi["LE"]:
            violations.append(f"{wi['id']}: LS+op_time > LE")

    if violations:
        print("  ✗ VIOLATIONS FOUND:")
        for v in violations:
            print(f"    {v}")
        raise ValueError("Time window validation failed!")
    else:
        print("  ✓ All time windows are consistent")

    # 9. Verify greedy solution feasibility
    print("Verifying greedy solution feasibility...")
    infeasible_count = 0

    for wid, (pickup_time, dropoff_time, scid) in schedule.items():
        wi = wi_by_id[wid]

        if pickup_time < wi["ES"] or pickup_time > wi["LS"]:
            infeasible_count += 1
            if pickup_time < wi["ES"]:
                wi["ES"] = int(pickup_time)
            if pickup_time > wi["LS"]:
                wi["LS"] = int(pickup_time)

        if dropoff_time < wi["EE"] or dropoff_time > wi["LE"]:
            if dropoff_time < wi["EE"]:
                wi["EE"] = int(dropoff_time)
            if dropoff_time > wi["LE"]:
                wi["LE"] = int(dropoff_time)

        if wi["LS"] + wi["op_time"] > wi["LE"]:
            wi["LE"] = wi["LS"] + wi["op_time"]

    if infeasible_count > 0:
        print(f"  ⚠ Adjusted {infeasible_count} windows")
    else:
        print(f"  ✓ Greedy solution is feasible")

    # 10. Build distance matrices
    sc_to_wi_dist = {}
    for sc in scs:
        for wi in wis:
            dist = manhattan_distance(sc["x"], sc["y"], wi["o_x"], wi["o_y"])
            sc_to_wi_dist[(sc["id"], wi["id"])] = dist

    wi_to_wi_dist = {}
    for w1 in wis:
        for w2 in wis:
            if w1["id"] != w2["id"]:
                dist = manhattan_distance(w1["d_x"], w1["d_y"], w2["o_x"], w2["o_y"])
                wi_to_wi_dist[(w1["id"], w2["id"])] = dist

    # 11. Write CSV
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["Horizon", horizon])
        writer.writerow(["SCs", num_scs])
        writer.writerow(["WIs", num_wis])
        writer.writerow([])

        writer.writerow(["type", "id", "x", "y", "start_time",
                         "o_x", "o_y", "d_x", "d_y",
                         "ES", "LS", "EE", "LE", "op_time"])

        for sc in scs:
            writer.writerow(["SC", sc["id"], f"{sc['x']:.2f}", f"{sc['y']:.2f}",
                             sc["start_time"],
                             "", "", "", "", "", "", "", "", ""])

        for wi in wis:
            writer.writerow(["WI", wi["id"], "", "", "",
                             f"{wi['o_x']:.2f}", f"{wi['o_y']:.2f}",
                             f"{wi['d_x']:.2f}", f"{wi['d_y']:.2f}",
                             wi["ES"], wi["LS"], wi["EE"], wi["LE"],
                             wi["op_time"]])

        writer.writerow([])

        writer.writerow(["src", "dst", "travel_time"])
        for (src, dst), time in sc_to_wi_dist.items():
            writer.writerow([src, dst, time])

        for (src, dst), time in wi_to_wi_dist.items():
            writer.writerow([src, dst, time])

    # Statistics
    avg_pickup_window = np.mean([w["LS"] - w["ES"] for w in wis])
    avg_dropoff_window = np.mean([w["LE"] - w["EE"] for w in wis])
    avg_op_time = np.mean([w["op_time"] for w in wis])

    print(f"\n{'=' * 60}")
    print(f"✓ Generated: {filename}")
    print(f"{'=' * 60}")
    print(f"  Horizon: {horizon}")
    print(f"  Avg operation time: {avg_op_time:.1f}")
    print(f"  Avg pickup window: {avg_pickup_window:.1f}")
    print(f"  Avg dropoff window: {avg_dropoff_window:.1f}")
    print(f"  Bottleneck WIs: {num_conflicts}")
    print(f"  Instance size factor: {num_wis / num_scs:.1f} WIs per SC")

    return {
        "routes": routes,
        "schedule": schedule,
        "scs": scs,
        "wis": wis,
        "horizon": horizon
    }


if __name__ == "__main__":
    os.makedirs("../Instances", exist_ok=True)

    test_cases = [
        (5, 30, "medium", 20),
        (5, 30, "hard", 20),
        (5, 30, "extreme", 20)
    ]

    for num_scs, num_wis, difficulty, seed in test_cases:
        print(f"\n{'=' * 80}")
        print(f"Generating {num_scs} SCs, {num_wis} WIs, difficulty={difficulty}, seed={seed}")
        print(f"{'=' * 80}")

        filename = f"Instances/final_{num_scs}SC_{num_wis}WI_{difficulty}_{seed}.csv"
        result = generate_instance(
            num_scs=num_scs,
            num_wis=num_wis,
            filename=filename,
            difficulty=difficulty,
            seed=seed
        )