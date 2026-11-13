#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export MAX_JOBS=1
method=${1:?method required}
stage=setup
failed() {
  code=$?
  python3 - "$method" "$stage" "$code" "$SLURM_JOB_ID" "$KC_SCRATCH" <<'PYFAIL'
import hashlib,json,platform,sys
from pathlib import Path
method,stage,code,job,scratch=sys.argv[1:]
h=hashlib.sha256()
for p in sorted(Path('src').glob('*')):
    if p.suffix in ('.py','.cpp','.cu','.h'):h.update(p.name.encode()+p.read_bytes())
Path(scratch,f'exact-failure-{method}-{job}.json').write_text(json.dumps(dict(method=method,stage=stage,exit_code=int(code),job=job,host=platform.node(),source_sha256=h.hexdigest()),indent=2)+'\n')
PYFAIL
  exit "$code"
}
trap failed ERR
short="$KC_SCRATCH/exact-validation-${method}-${SLURM_JOB_ID}-short.json"
long="$KC_SCRATCH/exact-validation-${method}-${SLURM_JOB_ID}-long.json"
case "$method" in
  smoke)
    stage=comprehensive_kernel_tests
    "$KC_PYTHON" test/experimental_kernels.py --output "$KC_SCRATCH/exact-kernel-test-${SLURM_JOB_ID}.json"
    stage=comprehensive_seeded_tests
    "$KC_PYTHON" test/seeded.py --device cuda
    stage=short_validation
    "$KC_PYTHON" test/exact_benchmark.py --device cuda --output "$short"
    stage=long_validation
    "$KC_PYTHON" test/exact_benchmark.py --device cuda --duration 4 --batch 1 --output "$long"
    stage=benchmark
    "$KC_PYTHON" benchmarks/exact_methods.py --quick --validation "$long" --validation-short "$short" --output "$KC_SCRATCH/exact-smoke-${SLURM_JOB_ID}.json"
    ;;
  cuda-adjoint-check)
    stage=comprehensive_adjoint_tests
    "$KC_PYTHON" test/adjoint.py
    stage=short_validation
    "$KC_PYTHON" test/exact_benchmark.py --device cuda --adjoint --method cpu-modal,cuda-modal,cuda-adjoint --output "$short"
    stage=long_validation
    "$KC_PYTHON" test/exact_benchmark.py --device cuda --adjoint --method cpu-modal,cuda-modal,cuda-adjoint --duration 4 --batch 1 --output "$long"
    stage=cpu_baseline
    "$KC_PYTHON" benchmarks/exact_methods.py --methods cpu-native,cpu-modal --batches 1 --durations .25,4 --validation "$long" --validation-short "$short" --output "$KC_SCRATCH/exact-adjoint-cpu-${SLURM_JOB_ID}.json"
    "$KC_PYTHON" benchmarks/exact_methods.py --methods cpu-modal --batches 4 --durations .25 --validation "$long" --validation-short "$short" --output "$KC_SCRATCH/exact-adjoint-cpu-b4-${SLURM_JOB_ID}.json"
    stage=benchmark
    "$KC_PYTHON" benchmarks/exact_methods.py --methods cuda-modal,cuda-adjoint --batches 1,4 --durations .25,4 --kinds vjp,loss_backward --repeats 3 --validation "$long" --validation-short "$short" --output "$KC_SCRATCH/exact-adjoint-check-${SLURM_JOB_ID}.json"
    ;;
  cpu-native|cpu-modal)
    stage=short_validation
    "$KC_PYTHON" test/exact_benchmark.py --device cpu --method "$method" --output "$short"
    stage=long_validation
    "$KC_PYTHON" test/exact_benchmark.py --device cpu --method "$method" --duration 4 --batch 1 --output "$long"
    stage=benchmark
    "$KC_PYTHON" benchmarks/exact_methods.py --methods "$method" --validation "$long" --validation-short "$short" --output "$KC_SCRATCH/exact-${method}-${SLURM_JOB_ID}.json"
    ;;
  cuda-modal|cuda-recompute|cuda-register|cuda-seeded|cuda-adjoint)
    extra=()
    if [ "$method" = cuda-adjoint ]; then extra+=(--adjoint); fi
    methods="cuda-modal,$method"
    if [ "$method" = cuda-modal ]; then methods=cuda-modal; fi
    case "$method" in
      cuda-recompute|cuda-register)
        stage=comprehensive_kernel_tests
        "$KC_PYTHON" test/experimental_kernels.py --output "$KC_SCRATCH/exact-kernel-test-${SLURM_JOB_ID}.json" ;;
      cuda-seeded)
        stage=comprehensive_seeded_tests
        "$KC_PYTHON" test/seeded.py --device cuda ;;
      cuda-adjoint)
        stage=comprehensive_adjoint_tests
        "$KC_PYTHON" test/adjoint.py ;;
    esac
    stage=short_validation
    "$KC_PYTHON" test/exact_benchmark.py --device cuda "${extra[@]}" --method "cpu-modal,$methods" --output "$short"
    stage=long_validation
    "$KC_PYTHON" test/exact_benchmark.py --device cuda "${extra[@]}" --method "cpu-modal,$methods" --duration 4 --batch 1 --output "$long"
    stage=cpu_baseline
    "$KC_PYTHON" benchmarks/exact_methods.py --methods cpu-native,cpu-modal --batches 1 --durations .25,4 --validation "$long" --validation-short "$short" --output "$KC_SCRATCH/exact-${method}-cpu-${SLURM_JOB_ID}.json"
    "$KC_PYTHON" benchmarks/exact_methods.py --methods cpu-modal --batches 4 --durations .25 --validation "$long" --validation-short "$short" --output "$KC_SCRATCH/exact-${method}-cpu-b4-${SLURM_JOB_ID}.json"
    if [ "$method" = cuda-register ]; then
      stage=benchmark
      "$KC_PYTHON" benchmarks/exact_methods.py --methods "$methods" --kinds synthesis --validation "$long" --validation-short "$short" --output "$KC_SCRATCH/exact-${method}-${SLURM_JOB_ID}.json"
    else
      stage=benchmark
      "$KC_PYTHON" benchmarks/exact_methods.py --methods "$methods" --validation "$long" --validation-short "$short" --output "$KC_SCRATCH/exact-${method}-${SLURM_JOB_ID}.json"
    fi
    ;;
  *) echo "Unknown method: $method" >&2; exit 2 ;;
esac
