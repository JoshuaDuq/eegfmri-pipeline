#!/usr/bin/env bash
# The in-scanner arm of the scanner / no-scanner comparison, end to end.
#
# Three stages in order, on subjects 0008 0010 0011 0015, at the same worker counts as
# the out-of-scanner run recorded in
# Données_Lorie_Ève/derivatives/logs/run_metadata/preprocessing/ (bad-channels n_jobs 2,
# ICA and epochs n_jobs 3). Each stage must succeed before the next starts.
#
#     bash studies/pain_study/scripts/run_inscanner_lorie_matched.sh
#
# Wrap it in caffeinate for an unattended run, or the Mac sleeps and the external drive
# unmounts mid-stage:
#
#     caffeinate -i -s bash studies/pain_study/scripts/run_inscanner_lorie_matched.sh

set -u -o pipefail

REPO="/Users/joduq24/Desktop/EEG_fMRI_Pipeline"
CFG="${REPO}/studies/pain_study/config/inscanner_lorie_matched.yaml"
CLI="${REPO}/.venv/bin/eeg-pipeline"
DERIV="/Volumes/KINGSTON/EEG_fMRI_data/derivatives_lorie_matched"
STAGED="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg_loriematched_staged"
LOGDIR="${DERIV}/logs/run_shell"
SUBJECTS=(--subject 0008 --subject 0010 --subject 0011 --subject 0015)

mkdir -p "${LOGDIR}"
STAMP="$(date +%Y%m%dT%H%M%S)"
MAIN_LOG="${LOGDIR}/run_${STAMP}.log"

say() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${MAIN_LOG}"; }

say "=== in-scanner arm, Lorie-matched parameters ==="
say "config      ${CFG}"
say "bids_root   ${STAGED}"
say "deriv_root  ${DERIV}"

# The whole run lives on the external drive. If it is not mounted, fail here rather than
# half way through ICA.
for path in "${STAGED}" "$(dirname "${DERIV}")"; do
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

run_stage bad-channels 2
run_stage ica          3
run_stage epochs       3

say "--- delivered"
for sub in sub-0008 sub-0010 sub-0011 sub-0015; do
  epo="${DERIV}/preprocessed/eeg/${sub}/eeg/${sub}_task-thermalactive_proc-clean_epo.fif"
  if [[ -f "${epo}" ]]; then
    say "  ${sub}: $(basename "${epo}") present"
  else
    say "  ${sub}: MISSING ${epo}"
  fi
done

say "=== done ==="
