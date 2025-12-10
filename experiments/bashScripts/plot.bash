#!/bin/bash

args=
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="${args} \"${i//\"/\\\"}\""
done

if [[ "${args}" == "" ]]; then args="/bin/bash"; fi

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

if [[ -e /scratch/work/public/singularity/greene-ib-slurm-bind.sh ]]; then
  source /scratch/work/public/singularity/greene-ib-slurm-bind.sh
fi

singularity exec ${nv} \
--overlay /scratch/projects/kaptanoglulab/MH/desc.ext3:ro \
--overlay /scratch/work/public/singularity/intel-oneapi-2022.1.2.sqf:ro \
--overlay /scratch/work/public/singularity/openmpi-4.1.2-ubuntu-22.04.1.sqf:ro \
/scratch/work/public/singularity/ubuntu-22.04.1.sif \
/bin/bash -c "
unset -f which
if [[ -e /ext3/apps/openmpi/4.1.2/env.sh ]]; then source /ext3/apps/openmpi/4.1.2/env.sh; fi
if [[ -e /ext3/env.sh ]]; then source /ext3/env.sh; fi
eval ${args}

python DESC/experiments/output_visualizer.py $1
"
