#!/bin/bash
# Run in AWS CloudShell. Shows progress; when the run is finished, prints the results and deletes
# everything launch.sh created (instance, bucket, role). Safe to run repeatedly.
set -uo pipefail
DIR=$HOME/pgw-bench
[ -f "$DIR/state" ] || { echo "No run found."; exit 0; }
eval "$(sed 's/ /\n/g' "$DIR/state")"
STATE=terminated
if [ -n "${IID:-}" ]; then
  STATE=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
    --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo terminated)
  echo "Instance $IID: $STATE"
  aws ec2 get-console-output --region "$REGION" --instance-id "$IID" --latest --output text \
    --query Output 2>/dev/null | grep -a "PGW-STEP" | tail -4
fi
if aws s3 cp --region "$REGION" "s3://$BUCKET/results.tgz" "$DIR/results.tgz" --only-show-errors 2>/dev/null; then
  mkdir -p "$DIR/results" && tar xzf "$DIR/results.tgz" -C "$DIR/results" 2>/dev/null
  echo; echo "===== RESULTS (copy everything between the lines) ====="
  cat "$DIR/results/out/summary.txt"
  echo "======================================================="
  CLEAN=1
elif [ "$STATE" = terminated ] || [ "$STATE" = shutting-down ]; then
  echo "The instance stopped without uploading results. Last log lines:"
  aws ec2 get-console-output --region "$REGION" --instance-id "$IID" --latest --output text --query Output 2>/dev/null | tail -30
  CLEAN=1
else
  echo "Still running; run collect.sh again in ~10 minutes."; exit 0
fi
if [ "${CLEAN:-0}" = 1 ]; then
  echo; echo "Cleaning up..."
  [ -n "${IID:-}" ] && aws ec2 terminate-instances --region "$REGION" --instance-ids "$IID" >/dev/null 2>&1
  aws s3 rb "s3://$BUCKET" --force --region "$REGION" >/dev/null 2>&1
  aws iam remove-role-from-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE" 2>/dev/null
  aws iam delete-instance-profile --instance-profile-name "$ROLE" 2>/dev/null
  aws iam delete-role-policy --role-name "$ROLE" --policy-name bucket 2>/dev/null
  aws iam delete-role --role-name "$ROLE" 2>/dev/null
  mv "$DIR/state" "$DIR/state.done.$(date +%s)"
  echo "Done: instance, bucket and role removed. Results kept in $DIR/results."
fi
