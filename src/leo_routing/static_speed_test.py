import time
import numpy as np
import numpy.typing as npt
from typing import Dict, Type, List
from dsns.helpers import SatID
from tqdm import tqdm
from argparse import ArgumentParser
import random

from dsns.solvers import DijkstraSolver, BmsspSolver, GraphSolver
from dsns.presets import MultiConstellation, StarlinkMultiConstellation
from dsns.constellation import WalkerISLHelper

solver_classes: Dict[str, Type[GraphSolver]] = {
    "bmssp": BmsspSolver,
    "dijkstra": DijkstraSolver,
}



def run_benchmark(
    solver_str: str,
    solver_class: Type[GraphSolver],
    constellation: MultiConstellation,
    ground_dishes: List[SatID],
    iterations: int,
) -> npt.NDArray[np.float64]:
    solver = solver_class()
    solver.update(constellation)

    QUERIES_PER_ITERATION = 100

    random.seed(42)
    test_pairs = [
        (random.choice(ground_dishes), random.choice(ground_dishes)) 
        for _ in range(QUERIES_PER_ITERATION)
    ]

    costs = np.ndarray((iterations, QUERIES_PER_ITERATION), dtype=float)

    desc = f"Running {solver_str} ({iterations} iters, {QUERIES_PER_ITERATION} pairs/iter)"
    
    t_start = time.time()
    for i in tqdm(range(iterations), desc=desc, unit="run"):
        for count, (u, v) in enumerate(test_pairs):
            cost = solver.benchmark_solve(u, v)
            costs[i, count] = cost

    t_end = time.time()
    total_time = t_end - t_start
    avg_per_iter = total_time / iterations
    avg_per_query = total_time / (iterations * QUERIES_PER_ITERATION)
    
    print(f"{solver_str} finished in {total_time:.4f}s")
    print(f"Avg time per iter: {avg_per_iter:.6f}s | Avg time per query: {avg_per_query:.6f}s")

    return costs


def validate_results(results: Dict[str, npt.NDArray[np.float64]]):
    solver_names = list(results.keys())

    matched = True
    for cost in results.values():
        if not np.isclose(results[solver_names[0]], cost, rtol=1e-9).all():
            matched = False
            break
    if matched:
        print("SUCCESS: solvers produce idential costs")
    else:
        print("FAILURE: solvers do not produce idential costs")


def verify_path(
    solver_str: str,
    solver_class: Type[GraphSolver],
    constellation: MultiConstellation,
    ground_dishes: List[SatID],
):
    solver = solver_class()
    solver.update(constellation)

    loading = tqdm(
        range(len(ground_dishes) ** 2), desc=f"Verifying paths for {solver_str}"
    )
    failures = 0
    for u in ground_dishes:
        for v in ground_dishes:
            cost = solver.get_path_cost(u, v)
            path = solver.get_path(u, v)
            if not verify_path_validity(solver.graph, path, cost):
                print("FAILURE, mismatch")
                failures += 1
            loading.update()

    if failures == 0:
        print("\nSUCCESS: All checks passed. BMSSP paths are valid and optimal.")
    else:
        print(f"\nFAILURE: {failures} checks failed.")


def verify_path_validity(graph, path: list[SatID], reported_cost: float) -> bool:
    if not path:
        return reported_cost == float("inf")

    calculated_cost = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        if u not in graph or v not in graph[u]:
            print(f"  [Error] Ghost edge detected: {u} -> {v} is not in the graph.")
            return False

        calculated_cost += graph[u][v]

    if not np.isclose(calculated_cost, reported_cost, rtol=1e-9):
        print(f"  [Error] Cost mismatch for path {path[0]}->{path[-1]}.")
        print(f"  Path Sum: {calculated_cost}")
        print(f"  Reported: {reported_cost}")
        return False

    return True

def main(
    solvers: list[str],
    iterations: int = 100,
    verify: bool = False,
    starting_planes: int = 18,
    starting_sats: int = 11,
    scales: int = 3,
):
    planes = starting_planes
    sats = starting_sats

    for scale in range(scales):
        print(f"\n{'='*60}")
        print(f"Benchmark Scale {scale + 1}/{scales}")
        print(f"Planes: {planes} | Sats/Plane: {sats} | Total Satellites: {planes * sats}")
        print(f"{'='*60}")

        isl_helper = WalkerISLHelper(
            num_planes=planes,
            sats_per_plane=sats,
            intra_layer_links=True,
            inter_layer_links=2,
        )

        starlink_kwargs = {
            "num_planes": planes,
            "sats_per_plane": sats,
            "isl_helper": isl_helper,
            "phase_offset": (65 / 72) * 360.0, # Keep standard offset logic
        }

        constellation = StarlinkMultiConstellation(starlink_kwargs=starlink_kwargs)
        constellation.update(1)
        ground_dishes = constellation.ground_constellation.satellites.ids

        results = {}

        for name in solvers:
            solver_class = solver_classes[name]
            if verify:
                verify_path(name, solver_class, constellation, ground_dishes)
            else:
                costs = run_benchmark(
                    name, solver_class, constellation, ground_dishes, iterations
                )
                results[name] = costs

        if not verify and len(solvers) > 1:
            validate_results(results)

        planes *= 2
        sats *= 2


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run static speed test comparing dijkstras and BMSSP with dynamic scaling"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations to repeat test, default=100",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=list(solver_classes.keys()),
        help="Type of solver to profile, if omitted runs all",
        required=False,
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check paths are valid, not to be used for CPU times as it makes use of caching",
    )
    # --- New scale arguments ---
    parser.add_argument(
        "--starting-planes",
        type=int,
        default=18,
        help="Number of planes in the first constellation scale",
    )
    parser.add_argument(
        "--starting-sats",
        type=int,
        default=11,
        help="Number of satellites per plane in the first scale",
    )
    parser.add_argument(
        "--scales",
        type=int,
        default=3,
        help="How many times to double and benchmark the constellation size",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.solver is None:
        solvers = list(solver_classes.keys())
    else:
        solvers = [args.solver]

    main(
        solvers, 
        args.iterations, 
        args.verify, 
        args.starting_planes, 
        args.starting_sats, 
        args.scales
    )
