#!/bin/bash
# Serblo Spot-H100 supervisor. Keeps the full pretraining run alive across spot preemptions:
# each "leg" is a Vertex CustomJob; when one is preempted/fails, resubmit; train.py resumes from
# GCS ckpt_latest.pt, so the LR schedule stays continuous across legs (and across alt accounts).
# Stops on SUCCEEDED (reached max_steps), or a stop-flag ($HOME/serblo_stop).
#
# CRITICAL: run only ONE instance, and do NOT run other gcloud concurrently; gcloud's shared
# credential store races under concurrency ("no active account"). This loop is serial by design.
# Re-entrant: on each cycle it adopts an already RUNNING/PENDING leg instead of duplicating.
#
#   REGION=us-central1 SPEC=/path/vertex_fullrun_spot.yaml nohup bash scripts/vertex_supervisor.sh &
set -uo pipefail
GC="--project=project-a9a36e4a-9735-4042-803"
REGION="${REGION:-us-central1}"
SPEC="${SPEC:?set SPEC to your Vertex CustomJob spec yaml}"
PN=895588934131
STOP="$HOME/serblo_stop"
LOG="$HOME/serblo_supervisor.log"
log(){ echo "$(date -u +%FT%TZ) SUP: $*" | tee -a "$LOG"; }

log "start (region=$REGION spec=$SPEC)"
leg=0
while [ ! -f "$STOP" ]; do
  jid=$(gcloud ai custom-jobs list --region="$REGION" $GC \
        --filter="displayName~serblo-full AND (state=JOB_STATE_RUNNING OR state=JOB_STATE_PENDING)" \
        --format="value(name.basename())" 2>/dev/null | head -1)
  if [ -n "$jid" ]; then
    log "adopting running leg $jid"
  else
    jid=$(gcloud ai custom-jobs create --region="$REGION" --display-name="serblo-full-l$leg" \
          --config="$SPEC" $GC 2>&1 | grep -o 'customJobs/[0-9]*' | head -1 | cut -d/ -f2)
    if [ -z "$jid" ]; then log "submit failed; retry 120s"; sleep 120; continue; fi
    log "leg $leg submitted $jid"; leg=$((leg + 1))
  fi
  while :; do
    [ -f "$STOP" ] && { log "STOP flag; leaving $jid running; exit"; exit 0; }
    st=$(gcloud ai custom-jobs describe "projects/$PN/locations/$REGION/customJobs/$jid" \
         --region="$REGION" $GC --format="value(state)" 2>/dev/null)
    case "$st" in
      *SUCCEEDED*)                 log "$jid SUCCEEDED — full run hit max_steps. DONE."; exit 0;;
      *FAILED*|*CANCEL*|*EXPIRED*) log "$jid ended ($st) — resubmit (resume from ckpt)"; sleep 15; break;;
      *)                           sleep 120;;
    esac
  done
done
log "stop flag present; exit"
