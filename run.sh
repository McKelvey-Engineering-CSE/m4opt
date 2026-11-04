
#!/bin/bash

OUTPUT="results.csv"

ALGORITHM="gcp"
PROBLEM="p1" # or p2

# Common command arguments
BASE_COMMAND=(
    schedule
    /home/punit/Documents/Telescope-Searching-Problem/data/ligo_healpix/GW200105_162426.fits
    SCHEDULE2.ecsv
    --mission=s30pro
    --nside=64
    --max-fields=100000
    --visits=1
    --delay=0s
    --deadline=1200s
    --time-step=1200s
    --absmag-mean=-21.5
    --no-appmag-dist
    --snr=3
    --exptime-min=1s
    --exptime-max=1200s
    --timelimit=1200s
    --jobs=0
    --initial-ra=41.5
    --initial-dec=-52
    --initial-roll=0.0
)

# CSV header
echo "time_ms,exit_code" > "$OUTPUT"

echo "================================"
echo "Running problem: $PROBLEM with algorithm: $ALGORITHM"
echo "================================"

# Start timer
start=$(date +%s%N)

# Run command
PYTHONPATH="/home/punit/Documents/Telescope-Searching-Overlap-Tiling/build/python${PYTHONPATH:+:$PYTHONPATH}" \
    uv run m4opt "${BASE_COMMAND[@]}" \
    --problem="$PROBLEM" \
    --cpp-algorithm=$ALGORITHM

exit_code=$?

# End timer
end=$(date +%s%N)

# Calculate elapsed milliseconds
elapsed=$(( (end - start) / 1000000 ))

# Record result
echo "$elapsed,$exit_code" >> "$OUTPUT"


echo "================================"
echo "Run complete!"
echo "Results saved to $OUTPUT"
