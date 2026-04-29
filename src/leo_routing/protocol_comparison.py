import os
import argparse
import pickle
import concurrent.futures
import numpy as np
from typing import Literal

from dsns.simulation import Simulation, LoggingActor
from dsns.solvers import NetworkItDijkstraSolver, DijkstraSolver
from dsns.message_actors import (
    GlobalRoutingActor,
    GlobalRoutingDataProvider,
    ResilientSourceRoutingActor,
    ResilientSourceRoutingDataProvider,
    SourceRoutingActor,
    SourceRoutingDataProvider,
    HardwareFailureActor,
)
from dsns.traffic_sim import MultiPointToPointTrafficActor, MultiPoissonTrafficActor
from dsns.presets import StarlinkMultiConstellation
from dsns.logging import PreprocessedLoggingActor, BandwidthLoggingActor
from dsns.transmission import LinkTransmissionActor


# GROUND_STATIONS = np.array(
#     [
#         (51.5072, -0.1276, 0.0),  # London
#         (40.7128, -74.0060, 0.0),  # New York
#         (52.5200, 13.4050, 0.0),  # Berlin
#         (31.2304, 121.4737, 0.0),  # Shanghai
#     ]
# )


GROUND_STATIONS = np.array(
    [
        (51.5072, -0.1276, 0.0),  # London
        (40.7128, -74.0060, 0.0),  # New York
        (-33.9249, 18.4241, 0.0),  # Cape Town
        (31.2304, 121.4737, 0.0),  # Shanghai
    ]
)

def run_sim(
    scenario_name: str,
    duration: float,
    pairs: Literal["long", "medium", "short", "all"],
    data_size: int,
    interval: float,
    model_bandwidth: bool,
    bandwidth: float,
    max_queue_size: int,
    failure_rate: float,
    recovery_rate: float,
    routing_data_provider_class: type[GlobalRoutingDataProvider] | type[SourceRoutingDataProvider] | type[ResilientSourceRoutingDataProvider],
    routing_actor_class: type[GlobalRoutingActor] | type[SourceRoutingActor] | type[ResilientSourceRoutingActor],
    routing_type_name: str = "Unknown Routing",
    poisson_traffic: bool = False,
    iteration: int = 1,
    update_interval: int = 1,
    logging: bool = True,
    results_dir: str = "results_dir/",
    verbose: bool = False,
):
    if verbose:
        print(f"--- Running {scenario_name} Simulation for {duration}s ({routing_type_name} - Iteration {iteration}) ---")
    constellation = StarlinkMultiConstellation(
        station_type="custom", ground_station_positions=GROUND_STATIONS
    )

    id_ldn = 0
    id_ny = 1
    id_ct = 2
    id_shng = 3

    message_config = [
        ("LND_NY", id_ldn, id_ny, data_size, interval),
        ("LND_CT", id_ldn, id_ct, data_size, interval),
        ("LND_SHNG", id_ldn, id_shng, data_size, interval),
    ]
    
    if pairs == "long":
        message_config = [message_config[2]]
    elif pairs == "medium":
        message_config = [message_config[1]]
    elif pairs == "short":
        message_config = [message_config[0]]

    actors = []
    data_providers = []

    if poisson_traffic:
        traffic_actor = MultiPoissonTrafficActor(
            message_config=message_config,
            update_interval=update_interval,
            cutoff=duration - 5
        )
    else:
        traffic_actor = MultiPointToPointTrafficActor(
            message_config=message_config, update_interval=update_interval
        )

    if failure_rate > 0.0:
        hw_failure_actor = HardwareFailureActor(failure_rate=failure_rate, recovery_rate=recovery_rate)
        actors.append(hw_failure_actor)

    actors.append(traffic_actor)

    routing_data_provider = routing_data_provider_class(solver=DijkstraSolver, update_interval=update_interval)
    

    if model_bandwidth:
        transmission_actor = LinkTransmissionActor(
            default_bandwidth=bandwidth,
            buffer_if_link_busy=True,
            max_queue_size=max_queue_size,
            reroute_threshold=0.5 if isinstance(routing_data_provider, ResilientSourceRoutingDataProvider) else None,
        )
        actors.append(transmission_actor)

    routing_actor = routing_actor_class(
        provider=routing_data_provider,
        update_interval=update_interval,
        model_bandwidth=model_bandwidth,
    )
    actors.append(routing_actor)
    data_providers.append(routing_data_provider)

    pre = PreprocessedLoggingActor(log_other=False) if logging else None
    bw_logger = BandwidthLoggingActor() if (logging and model_bandwidth) else None
    
    logging_actors = []
    if logging:
        logging_actors.append(pre)
        if bw_logger:
            logging_actors.append(bw_logger)
    if verbose:
        logging_actors.append(LoggingActor(verbose=True))

    sim = Simulation(
        constellation,
        actors=actors,
        logging_actors=logging_actors,
        data_providers=data_providers,
        timestep=1,
    )

    init_time = 0
    sim.initialize(init_time)
    sim.run(duration, progress=iteration == 1)

    if logging and pre:
        os.makedirs(results_dir, exist_ok=True)
        filename = os.path.join(
            results_dir, f"{scenario_name}_{pairs}_{routing_type_name}_{duration}s_{iteration}_iter{'bw' if model_bandwidth else 'no_bw'}.pickle"
        )
        
        payload = [pre.direct_messages, pre.broadcast_messages, pre.other_events]
        if bw_logger:
            bw_stats = bw_logger.aggregate(period=1.0, default_bandwidth=bandwidth)
            payload.append(bw_stats)
            
        with open(filename, "wb") as f:
            pickle.dump(tuple(payload), f)
        if verbose:
            print(f"[{routing_type_name} Iteration {iteration}] Results logged to {filename}")


def main(
    scenario_name: str,
    iterations: int,
    duration: float,
    pairs: Literal["long", "medium", "short", "all"],
    data_size: int,
    interval: float,
    model_bandwidth: bool,
    bandwidth: float,
    max_queue_size: int,
    failure_rate: float,
    recovery_rate: float,
    poisson_traffic: bool,
    logging: bool = True,
    results_dir: str = "results_dir/",
    verbose: bool = False,
):
    routing_actors = {
        "GlobalRouting": (GlobalRoutingDataProvider, GlobalRoutingActor),
        "SourceRouting": (SourceRoutingDataProvider, SourceRoutingActor),
        "ResilientRouting": (ResilientSourceRoutingDataProvider, ResilientSourceRoutingActor),
    }

    max_workers = os.cpu_count() or 4
    if verbose:
        print(f"Starting parallel execution with {max_workers} concurrent processes...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for routing_type, (routing_data_provider, routing_actor) in routing_actors.items():
            for i in range(iterations):
                future = executor.submit(
                    run_sim,
                    scenario_name=scenario_name,
                    duration=duration,
                    pairs=pairs,
                    data_size=data_size,
                    interval=interval,
                    model_bandwidth=model_bandwidth,
                    bandwidth=bandwidth,
                    max_queue_size=max_queue_size,
                    failure_rate=failure_rate,
                    recovery_rate=recovery_rate,
                    routing_data_provider_class=routing_data_provider,
                    routing_actor_class=routing_actor,
                    routing_type_name=routing_type,
                    poisson_traffic=poisson_traffic,
                    iteration=i + 1,
                    update_interval=1,
                    logging=logging,
                    results_dir=results_dir,
                    verbose=verbose,
                )
                futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"A simulation run generated an exception: {exc}")


def get_parser():
    parser = argparse.ArgumentParser("Unified LEO Routing Simulation Test Runner")
    
    # General Config
    parser.add_argument("--scenario-name", type=str, default="unified_scenario", help="Name to prefix output files with")
    parser.add_argument("--duration", type=float, default=300, help="Duration of each simulation run in seconds")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations to repeat test")
    parser.add_argument("--pairs", type=str, choices=["long", "medium", "short", "all"], default="all", help="Which ground station pairs to test (long, medium, short, all)")    
    # Traffic Config
    parser.add_argument("--data-size", type=int, default=256, help="Volume of individual data payloads for stream (Bytes)")
    parser.add_argument("--interval", type=float, default=1.0, help="Frequency between generation of messages (seconds)")
    
    # Bandwidth controls
    parser.add_argument("--model-bandwidth", action="store_true", help="Enable bandwidth and queuing models (Baseline Efficiency Mode)")
    parser.add_argument("--bandwidth", type=float, default=100_000, help="Default bandwidth for links (bytes/s)")
    parser.add_argument("--max-queue-size", type=int, default=500, help="Maximum buffer queue size before dropping")
    
    # Failure controls
    parser.add_argument("--failure-rate", type=float, default=0.0, help="Probability of a link failing per second (Set > 0 for Robustness/Chaos Mode)")
    parser.add_argument("--recovery-rate", type=float, default=0.05, help="Probability of a failed link recovering per second")
    
    parser.add_argument("--poisson-traffic", action="store_true", help="Use Poisson traffic generation instead of fixed interval (only valid if --pairs is not 'all')")

    # Output Setup
    parser.add_argument("-l", "--logging", action="store_true", help="Enable logging of results", default=True)
    parser.add_argument("--results-dir", type=str, default="results/unified_sim/")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()

    main(
        scenario_name=args.scenario_name,
        iterations=args.iterations,
        pairs=args.pairs,
        duration=args.duration,
        data_size=args.data_size,
        interval=args.interval,
        model_bandwidth=args.model_bandwidth,
        bandwidth=args.bandwidth,
        max_queue_size=args.max_queue_size,
        failure_rate=args.failure_rate,
        recovery_rate=args.recovery_rate,
        logging=args.logging,
        results_dir=args.results_dir,
        verbose=args.verbose,
        poisson_traffic=args.poisson_traffic,
    )
