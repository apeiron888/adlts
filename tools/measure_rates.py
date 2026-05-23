#!/usr/bin/env python3
"""
Measure enqueue vs processing rates by polling backend /stats endpoint.
Usage: python tools/measure_rates.py [duration_seconds]
Default duration: 20s
"""
import sys, time, json
from urllib.request import urlopen, Request
from urllib.error import URLError

URL = 'http://localhost:8000/stats'

def fetch_stats():
    try:
        req = Request(URL, headers={'Accept': 'application/json'})
        with urlopen(req, timeout=5) as r:
            return json.load(r)
    except URLError as e:
        print(f"ERROR: cannot reach {URL}: {e}")
        return None


def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    interval = 1.0
    samples = int(duration / interval)

    prev = None
    results = []
    print(f"Polling {URL} every {interval}s for {duration}s...")
    for i in range(samples+1):
        stats = fetch_stats()
        t = time.time()
        if stats is None:
            time.sleep(interval)
            continue
        # Extract values with safe defaults
        sr = stats.get('stream_receiver', {})
        q = stats.get('queue', {})
        qc = stats.get('queue_consumer', {})
        frames_received = sr.get('frames_received', 0)
        frames_to_queue = sr.get('frames_to_queue', 0)
        queue_depth = q.get('depth', 0)
        dropped = q.get('dropped', 0)
        frames_processed = qc.get('frames_processed', 0)

        point = {
            't': t,
            'frames_received': frames_received,
            'frames_to_queue': frames_to_queue,
            'queue_depth': queue_depth,
            'dropped': dropped,
            'frames_processed': frames_processed,
        }
        if prev is not None:
            dt = t - prev['t']
            if dt <= 0:
                dt = interval
            prod_rate = (frames_to_queue - prev['frames_to_queue']) / dt
            recv_rate = (frames_received - prev['frames_received']) / dt
            cons_rate = (frames_processed - prev['frames_processed']) / dt
            print(f"[{i}] recv={recv_rate:.2f}/s  queued={prod_rate:.2f}/s  proc={cons_rate:.2f}/s  depth={queue_depth}  dropped={dropped}")
            results.append((recv_rate, prod_rate, cons_rate, queue_depth, dropped))
        else:
            print(f"[0] frames_received={frames_received}  frames_to_queue={frames_to_queue}  frames_processed={frames_processed}  depth={queue_depth}  dropped={dropped}")
        prev = point
        time.sleep(interval)

    # Summarize
    if results:
        avg_recv = sum(r[0] for r in results)/len(results)
        avg_queued = sum(r[1] for r in results)/len(results)
        avg_proc = sum(r[2] for r in results)/len(results)
        max_depth = max(r[3] for r in results)
        total_dropped = results[-1][4] if results else 0
        print('\nSummary over polling period:')
        print(f"  avg recv rate:  {avg_recv:.2f} fps")
        print(f"  avg queue in:   {avg_queued:.2f} fps")
        print(f"  avg proc rate:  {avg_proc:.2f} fps")
        print(f"  max queue depth: {max_depth}")
        print(f"  dropped (end):   {total_dropped}")
    else:
        print('No samples collected.')

if __name__ == '__main__':
    main()
