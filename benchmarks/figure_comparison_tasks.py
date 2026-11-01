"""Write a short-job manifest; keep cached curves in one allocation."""
import json
import sys

BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
tasks = [{'id': 'kc-figure-00-caches', 'command': ['bash', 'hpc/figure-comparison-shard.sh',
          'cached3,cached1', ','.join(map(str, BATCHES))], 'estimated_seconds': 1800}]
for prefix, method in (('01', 'diaz'), ('02', 'fdtd')):
    for batch in BATCHES:
        tasks.append({'id': f'kc-figure-{prefix}-{method}-b{batch:03d}',
                      'command': ['bash', 'hpc/figure-comparison-shard.sh', method, str(batch)],
                      'estimated_seconds': 1800})
json.dump(tasks, sys.stdout, indent=2)
print()
