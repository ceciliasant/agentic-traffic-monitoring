# setup
docker compose build

# agentic - run the coordinator to generate trace.jsonl
docker compose run --rm --entrypoint sh coordinator -c "rm -f /app/output/trace.jsonl"

docker compose run --rm coordinator

# classic 
docker compose run --rm --entrypoint python coordinator classic_runner.py

# comparison
docker compose run --rm --entrypoint python coordinator compare.py
cat /app/output/comparison/summary.md

# pick a window for comparison 
$ docker compose run --rm -v ./pick_window.py:/app/pick_window.py --entrypoint python coordinator pick_window.py 2024-07-01 2024-09-30

# delete the logs and outpus
docker compose run --rm --entrypoint sh coordinator -c "rm -f /app/output/trace.jsonl /app/output/comparison/agent_predictions.json"
docker compose run --rm --entrypoint sh coordinator -c "ls -l /app/output /app/output/comparison"

# delete ditto predictions - python script from postflows

# run the test week
nohup systemd-inhibit --what=sleep:idle --why="overnight run" ./run_week.sh > /dev/null 2>&1 &
[1] 129467

## Check the status of the comparison
cat results/progress.log          # which steps finished
grep -c "^\[classic\]" results/classic.log
grep -- "- Hour:" results/agent.log | tail -3
cat results/summary.md            # the comparison, if it got that far

## docker compose build
##