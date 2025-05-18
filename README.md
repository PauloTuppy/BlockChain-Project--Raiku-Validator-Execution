# Raiku Blockchain Validator Implementation

A high-performance blockchain validator implementation with BFT consensus, execution engine, and comprehensive benchmarking.

## Project Structure

- `bft_consensus/`: Byzantine Fault Tolerant consensus implementation
- `execution_node/`: Transaction execution engine
- `shared_state_sync/`: State synchronization between nodes
- `proactive_scheduler/`: Transaction scheduling optimization
- `zk_prover/`: Zero-knowledge proof components
- `sidecar/`: Monitoring and auxiliary services
- `benchmark/`: Performance testing and analysis tools
  - `load_test/`: Locust-based load testing
  - `monitoring/`: Prometheus/Grafana monitoring
  - `aws_deploy/`: Terraform deployment scripts

## Key Features

- High-throughput transaction processing
- Optimized consensus algorithm
- Comprehensive monitoring and metrics
- Automated benchmarking framework
- Cloud deployment ready

## Getting Started

### Prerequisites

- Rust (latest stable)
- Python 3.10+ (for benchmarking tools)
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository
2. Build the validator:
```bash
cargo build --release
```

3. Install benchmarking dependencies:
```bash
cd benchmark
pip install -r requirements.txt
```

## Running the Validator

```bash
./target/release/execution_node --config config.toml
```

## Benchmarking

To run performance tests:
```bash
cd benchmark
./run_benchmark.sh
```

View results:
```bash
python analyze_results.py
```

## Monitoring

The validator exposes Prometheus metrics on port `9090`. Preconfigured dashboards are available in `benchmark/monitoring/`.

## License

MIT
