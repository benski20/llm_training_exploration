#!/bin/bash
# Run in AWS CloudShell (us-east-2), after uploading pgw-cpu-bench.tar.gz, launch.sh and bench.sh.
# Starts ONE CPU instance (default c7i.16xlarge, 64 vCPUs, ~$2.9/h) that benchmarks and then terminates itself.
set -euo pipefail
REGION=${REGION:-us-east-2}
TYPE=${TYPE:-c7i.16xlarge}
TAR=${TAR:-$HOME/pgw-cpu-bench.tar.gz}
HERE=$(cd "$(dirname "$0")" && pwd)
DIR=$HOME/pgw-bench; mkdir -p "$DIR"
[ -f "$TAR" ] || { echo "Missing $TAR: upload pgw-cpu-bench.tar.gz first (Actions > Upload file)."; exit 1; }
[ ! -f "$DIR/state" ] || { echo "A run already exists ($DIR/state). Run collect.sh first."; exit 1; }

ACCT=$(aws sts get-caller-identity --query Account --output text)
STAMP=$(date +%s)
BUCKET=pgw-bench-$ACCT-$STAMP
ROLE=pgw-bench-role-$STAMP
echo "Private bucket $BUCKET"
aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
  --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
aws s3 cp "$TAR" "s3://$BUCKET/pgw.tgz" --region "$REGION" --only-show-errors

echo "Instance role $ROLE (read/write this bucket only)"
aws iam create-role --role-name "$ROLE" --assume-role-policy-document \
  '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
aws iam put-role-policy --role-name "$ROLE" --policy-name bucket --policy-document \
  "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\"],\"Resource\":\"arn:aws:s3:::$BUCKET/*\"}]}"
aws iam create-instance-profile --instance-profile-name "$ROLE" >/dev/null
aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
echo "REGION=$REGION BUCKET=$BUCKET ROLE=$ROLE IID=" > "$DIR/state"
sleep 15   # IAM propagation

AMI=$(aws ssm get-parameter --region "$REGION" \
  --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
  --query Parameter.Value --output text)
sed -e "s|__BUCKET__|$BUCKET|" -e "s|__REGION__|$REGION|" "$HERE/bench.sh" > "$DIR/user-data.sh"

IID=""
for i in 1 2 3 4; do
  IID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" \
    --iam-instance-profile Name="$ROLE" --instance-initiated-shutdown-behavior terminate \
    --metadata-options HttpTokens=required \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":60,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --user-data "file://$DIR/user-data.sh" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=pgw-cpu-bench}]' \
    --query 'Instances[0].InstanceId' --output text 2>"$DIR/run.err") && break
  grep -q "Invalid IAM Instance Profile" "$DIR/run.err" && { echo "waiting for IAM..."; sleep 20; continue; }
  cat "$DIR/run.err"; exit 1
done
[ -n "${IID:-}" ] || { echo "Could not start the instance; run collect.sh to clean up."; exit 1; }
echo "REGION=$REGION BUCKET=$BUCKET ROLE=$ROLE IID=$IID" > "$DIR/state"
echo
echo "Started $IID ($TYPE). It installs, benchmarks (~45-75 min) and terminates itself (hard cap 2.5 h)."
echo "Check progress any time with:  bash collect.sh"
