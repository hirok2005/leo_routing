from typing import List
from tqdm import tqdm
from argparse import ArgumentParser

from dsns.helpers import SatID
from dsns.encoders import source_route_encode, source_route_decode, source_route_read_next_segment
from dsns.solvers import DijkstraSolver
from dsns.presets import MultiConstellation, StarlinkMultiConstellation


def run_benchmark(constellation: MultiConstellation, ground_dishes: List[SatID]):
    solver = DijkstraSolver()
    solver.update(constellation)

    desc = "Running Test"

    incorrect = 0
    tot = len(ground_dishes) ** 2

    for u in ground_dishes:
        for v in ground_dishes:
            if u == v:
                continue
            path = solver.get_path(u, v)
            print(path)
            encoded = source_route_encode(path, constellation)
            decoded = source_route_decode(encoded, u, constellation)
            # if encoded != decoded:
            #     print(encoded)
            #     print(decoded)
            #     print()
            incorrect += path != decoded

    if incorrect == 0:
        print("SUCCESS: encoder and decoder are correct")
    else:
        print("FAILURE: encoder and decoder are incorrect")


# def validate_results(results: Dict[str, npt.NDArray[np.float64]]):
#     solver_names = list(results.keys())
#
#     matched = True
#     for cost in results.values():
#         if not np.isclose(results[solver_names[0]], cost, rtol=1e-9).all():
#             matched = False
#             break
#     if matched:
#         print("SUCCESS: solvers produce idential costs")
#     else:
#         print("FAILURE: solvers do not produce idential costs")


def main():
    constellation = StarlinkMultiConstellation()
    constellation.update(0)
    ground_dishes = constellation.starlink_constellation.satellites.ids

    run_benchmark(constellation, ground_dishes)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run source routing encoder test on Starlink constellation"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    main()
