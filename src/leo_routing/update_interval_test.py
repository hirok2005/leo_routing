import os
import pickle
import random
import datetime
import time
from typing import List, Optional
import argparse

from dsns.logging import BandwidthLoggingActor, PreprocessedLoggingActor
from dsns.message import Link
from dsns.simulation import Simulation, LoggingActor
from dsns.message_actors import (
    GlobalRoutingActor,
    GlobalRoutingDataProvider,
    SourceRoutingActor,
    SourceRoutingDataProvider,
)
from dsns.traffic_sim import MultiPointToPointTrafficActor
from dsns.presets import StarlinkMultiConstellation


def main(
    values: list[float],
    duration: float,
    iterations: int = 100,
    logging: bool = True,
    results_dir: str = "results_dir/",
    verbose: bool = False,
):
    time_taken = []

    if logging:
        os.makedirs(results_dir, exist_ok=True)

    for update_interval in values:
        t_start = time.time()
        for i in range(iterations):
            print(
                f"Update Interval: {update_interval}s - Iteration {i + 1}/{iterations}"
            )
            run_sim(
                update_interval,
                duration,
                logging=logging,
                results_dir=results_dir,
                verbose=verbose,
            )

        elapsed = time.time() - t_start
        time_taken.append(elapsed)
        print(f"Completed interval {update_interval}s in {elapsed:.2f}s")


def run_sim(
    update_interval: float,
    duration: float,
    logging: bool = True,
    results_dir: str = "results_dir/",
    verbose: bool = False,
):
    constellation = StarlinkMultiConstellation()
    ground_ids = constellation.ground_constellation.satellites.ids

    try:
        bahrain = ground_ids[2]
        hawaii = ground_ids[7]
        sing = ground_ids[5]
        seoul = ground_ids[4]

        message_config = [
            ("Short_IE_SE", sing, seoul, 10, 5),
            ("Med_IE_US", bahrain, sing, 10, 5),
            ("Long_US_AU", bahrain, hawaii, 10, 5),
        ]
    except IndexError:
        print("Warning: Not enough ground stations found. Falling back to first/last.")
        message_config = [("Default", ground_ids[0], ground_ids[-1], 1000, 1.0)]

    actors = []
    data_providers = []

    traffic_actor = MultiPointToPointTrafficActor(
        message_config=message_config, update_interval=1.0
    )

    actors.append(traffic_actor)

    # routing_data_provider = SourceRoutingDataProvider(update_interval=update_interval)
    routing_data_provider = GlobalRoutingDataProvider(update_interval=update_interval)

    # routing_actor = SourceRoutingActor(
    #     provider=routing_data_provider,
    #     update_interval=update_interval,
    #     model_bandwidth=False,
    # )
    routing_actor = GlobalRoutingActor(
        provider=routing_data_provider,
        update_interval=update_interval,
        model_bandwidth=False,
    )

    actors.append(routing_actor)
    data_providers.append(routing_data_provider)

    if logging:
        pre = PreprocessedLoggingActor(log_other=False)
        logging_actors = [pre]
    else:
        logging_actors = []

    sim = Simulation(
        constellation,
        actors=actors,
        logging_actors=logging_actors
        + ([LoggingActor(verbose=True)] if verbose else []),
        data_providers=data_providers,
        timestep=1,
    )

    sim.initialize(0)
    sim.run(duration, progress=True)

    if logging:
        now = f"{datetime.datetime.now(): %Y-%m-%d_%H-%M-%S}"
        with open(
            os.path.join(
                results_dir, f"update_inteval_{update_interval}_{duration}_{now}.pickle"
            ),
            "wb",
        ) as f:
            pickle.dump(
                (pre.direct_messages, pre.broadcast_messages, pre.other_events), f
            )


def get_parser():
    parser = argparse.ArgumentParser("Routing Update Interval Test")
    parser.add_argument(
        "--values",
        type=float,
        nargs="+",
        default=[1, 5, 10, 15, 20, 50, 100],
        help="List of values to test update_interval (in seconds)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60 * 60 * 2,
        help="Duration of each run, in seconds",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of iterations to repeat test per value",
    )
    parser.add_argument("-l", action="store_true", help="Enable logging")
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("-v", action="store_true", help="Enable verbose output")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args.values, args.duration, args.iterations, args.l, args.results_dir, args.v)
