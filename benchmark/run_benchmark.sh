#!/bin/bash

# Benchmark configuration
TPS_START=2000
TPS_END=5000
TPS_STEP=1000
DURATION_MIN=15

# Initialize results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/$TIMESTAMP"
mkdir -p $RESULTS_DIR

# Run benchmark at different TPS levels
for (( tps=$TPS_START; tps<=$TPS_END; tps+=$TPS_STEP )); do
    echo "Running benchmark at $tps TPS for $DURATION_MIN minutes"
    
    # Start load test
    locust -f load_test/locustfile.py \
        --headless \
        --users $tps \
        --spawn-rate $tps \
        --run-time ${DURATION_MIN}m \
        --csv $RESULTS_DIR/load_${tps}tps \
        --html $RESULTS_DIR/report_${tps}tps.html &
    
    # Monitor system metrics
    prometheus --config.file=monitoring/prometheus.yml \
        --storage.tsdb.path=$RESULTS_DIR/prom_data_${tps}tps &
    
    sleep $(($DURATION_MIN * 60))
    
    # Stop services
    pkill -f locust
    pkill -f prometheus
    sleep 10
    
    # Collect and analyze results
    python3 analyze_results.py $RESULTS_DIR $tps
done

# Generate final report
python3 generate_report.py $RESULTS_DIR
