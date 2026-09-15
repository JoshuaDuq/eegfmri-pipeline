#!/usr/bin/env bash
# Preprocess subjects on one arm of the simulator/scanner comparison, end to end.
#
# Three stages in order, at the same worker counts as the run that preprocessed
# 0008/0010/0011/0015 (bad-channels n_jobs 2, ICA and epochs n_jobs 3). Each stage must
# succeed before the next starts.
#
#     bash studies/pain_study/scripts/run_simulator_arm.sh 0019 0020 0021
#     bash studies/pain_study/scripts/run_simulator_arm.sh \
#         --config studies/pain_study/config/fastr_bcgnet_arm.yaml 0019 0020 0021
#     bash studies/pain_study/scripts/run_simulator_arm.sh --stages epochs 0019 0020 0021
#
# Run it from a COPY, not from the repo, if there is any chance the repo file will be
# edited while it runs. Bash reads a script incrementally by byte offset, so rewriting the
# file underneath a running instance makes it resume mid-token; that is what killed the
# 2026-09-03 simulator run between its ICA and epochs stages, after both expensive stages
# had already succeeded.
#
# Wrap it in caffeinate for an unattended run, or the Mac sleeps and the external drive
# unmounts mid-stage:
#
#     caffeinate -i -s bash studies/pain_study/scripts/run_simulator_arm.sh 0019 0020 0021
#
# Conversion is not part of this script. Both arms' recordings live flat -- the simulator
# ones under source_data_sim/sub_NNNN/eeg/, the FASTR+BCGNet ones under
# fastr_python_v2_bcgnet/sub-NNNN/ -- and the converter globs sub-*/eeg/<layout>/, so a new
# subject needs a staged symlink tree first.

set -u -o pipefail

REPO="/Users/joduq24/Desktop/EEG_fMRI_Pipeline"
CLI="${REPO}/.venv/bin/eeg-pipeline"
CFG="${REPO}/studies/pain_study/config/simulator_arm.yaml"

STAGES="bad-channels,ica,epochs"

while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --config) CFG="$2"; shift 2 ;;
    # Resume a run that died part way through. The stages already done are on disk and
    # their inputs are cached, so naming only what is left costs nothing and skips hours.
    --stages) STAGES="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "usage: $0 [--config PATH] [--stages bad-channels,ica,epochs] <subject> [subject ...]" >&2
  exit 2
fi
if [[ ! -f "${CFG}" ]]; then
  echo "ABORT: no such config: ${CFG}" >&2
  exit 2
fi

# Read the roots out of the config rather than repeating them here, so an arm is defined
# in exactly one place.
read -r BIDS DERIV < <("${CLI}" --config "${CFG}" info config --json \
  | "${REPO}/.venv/bin/python" -c \
    'import json,sys; c=json.load(sys.stdin); print(c["bids_root"], c["deriv_root"])')
if [[ -z "${BIDS}" || -z "${DERIV}" ]]; then
  echo "ABORT: could not read bids_root/deriv_root from ${CFG}" >&2
  exit 1
fi
LOGDIR="${DERIV}/logs/run_shell"

SUBJECTS=()
for sub in "$@"; do
  SUBJECTS+=(--subject "${sub}")
done

mkdir -p "${LOGDIR}"
STAMP="$(date +%Y%m%dT%H%M%S)"
MAIN_LOG="${LOGDIR}/run_${STAMP}.log"

say() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${MAIN_LOG}"; }

say "=== $(basename "${CFG}" .yaml) ==="
say "config      ${CFG}"
say "bids_root   ${BIDS}"
say "deriv_root  ${DERIV}"
say "subjects    $*"

# The whole run lives on the external drive. If it is not mounted, fail here rather than
# half way through ICA.
# deriv_root itself is created by the first stage, so its parent is what must exist.
for path in "${BIDS}" "$(dirname "${DERIV}")"; do
  if [[ ! -d "${path}" ]]; then
    say "ABORT: ${path} is not present. Is /Volumes/KINGSTON mounted?"
    exit 1
  fi
done

run_stage() {
  local stage="$1"; shift
  local njobs="$1"; shift
  local log="${LOGDIR}/${STAMP}_${stage}.log"
  say "--- ${stage} (n_jobs=${njobs}) -> ${log}"
  local started="${SECONDS}"
  local rc=0
  "${CLI}" --config "${CFG}" preprocessing "${stage}" \
      "${SUBJECTS[@]}" --n-jobs "${njobs}" >>"${log}" 2>&1 || rc=$?
  local mins=$(( (SECONDS - started) / 60 ))
  if [[ ${rc} -ne 0 ]]; then
    say "FAILED: ${stage} exited ${rc} after ${mins} min. Tail of ${log}:"
    tail -n 40 "${log}" | tee -a "${MAIN_LOG}"
    exit "${rc}"
  fi
  say "ok: ${stage} in ${mins} min"
}

# A case, not an associative array: macOS ships bash 3.2, where `declare -A` quietly
# makes an INDEXED array and then arithmetic-evaluates the key, so STAGE_JOBS[bad-channels]
# dies on `bad: unbound variable` under `set -u`.
# Defaults reproduce the worker counts of the runs that built the existing derivatives.
# Override per stage with the environment variables when wall-clock matters: the counts are
# a scheduling knob, not a science one -- each subject's ICA is seeded from
# project.random_state and does not depend on how many run alongside it.
#
# What the counts buy differs by stage. bad-channels parallelises over FILES (42 on the
# seven-subject FASTR arm) and holds one run per worker, so it scales well. ICA and epochs
# parallelise over SUBJECTS, so wall time is ceil(n_subjects / workers) x per-subject time:
# on 7 subjects, 4/5/6 workers are all two waves and only 7+ is one, while each ICA worker
# wants ~3 GB. More workers than subjects does nothing at all.
stage_jobs() {
  case "$1" in
    bad-channels) echo "${BAD_CHANNELS_JOBS:-2}" ;;
    ica)          echo "${ICA_JOBS:-3}" ;;
    epochs)       echo "${EPOCHS_JOBS:-3}" ;;
    *)            echo "" ;;
  esac
}

say "stages      ${STAGES}"
IFS=',' read -r -a REQUESTED <<< "${STAGES}"
for stage in "${REQUESTED[@]}"; do
  if [[ -z "$(stage_jobs "${stage}")" ]]; then
    say "ABORT: unknown stage '${stage}'; expected bad-channels, ica or epochs"
    exit 2
  fi
done
for stage in "${REQUESTED[@]}"; do
  run_stage "${stage}" "$(stage_jobs "${stage}")"
done

say "--- delivered"
for sub in "$@"; do
  epo="${DERIV}/preprocessed/eeg/sub-${sub}/eeg/sub-${sub}_task-thermalactive_proc-clean_epo.fif"
  if [[ -f "${epo}" ]]; then
    say "  sub-${sub}: $(basename "${epo}") present"
  else
    say "  sub-${sub}: MISSING ${epo}"
  fi
done

say "=== done ==="
