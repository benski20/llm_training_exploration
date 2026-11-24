# AWS CPU benchmarks (customer side)

Measures the customer gateway's CPU work on real AWS hardware, paid from AWS credits: the per-token
gateway CPU floor at 7B / 32B / 72B widths, the pks-v3 mask kernels, and the customer CPU per mask slot
(plain / mlwe / pks-v3) at 7B and 32B. No model download, no GPU.

1. `./make_tarball.sh` (here) -> `pgw-cpu-bench.tar.gz` (source only).
2. AWS Console, region **US East (Ohio)**, open **CloudShell**; Actions > Upload file: the tarball,
   `launch.sh`, `bench.sh`, `collect.sh`.
3. `bash launch.sh` - creates a private S3 bucket, a role that can only read/write that bucket, and one
   `c7i.16xlarge` (64 vCPUs, about $2.9/h) that benchmarks and terminates itself (hard cap 2.5 h).
4. After about an hour: `bash collect.sh` - prints the results and deletes the instance, bucket and role.
   Run it any time before that to see progress. `TYPE=c7i.8xlarge bash launch.sh` for a 32-vCPU box.
