#!/bin/bash
base_dir=$(pwd)
run_desc_script="${base_dir}/run-desc.bash"
cd ${base_dir}
echo ${base_dir}
rm -rf *.ext3 Miniconda3-*.sh tmp-*.bash run-desc.bash DESC/
cd ${base_dir}
cp -rp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz . && gunzip overlay-15GB-500K.ext3.gz && mv overlay-15GB-500K.ext3 desc.ext3
module purge && module load singularity-ce/4.3.3
echo "module purge && module load singularity-ce/4.3.3" > ${run_desc_script}
echo "singularity exec --nv --overlay ${base_dir}/desc.ext3:rw /share/apps/images/ubuntu-24.04.3.sif /bin/bash -c \" " >> ${run_desc_script}
echo "source /ext3/env.sh && \ " >> ${run_desc_script}
echo "eval /bin/bash" >> ${run_desc_script}
echo " \" " >> ${run_desc_script}
chmod 755 ${run_desc_script}

tmp_script="setup_conda_environment.bash"
tmp_setup_conda="setup-conda.bash"
cp -rp /share/apps/utils/setup-conda.bash ${tmp_setup_conda}
cat<<EOF | tee ${tmp_script}
bash ${tmp_setup_conda}
cat<<EOF2 >> /ext3/env.sh
source /opt/apps/lmod/lmod/init/sh
export MODULEPATH=/ext3/apps/modulefiles
module load mkl/2022.0.2
EOF2
source /ext3/env.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
rm -rf /ext3/miniconda3/compiler_compat/ld
conda create -p /ext3/py3.12 python=3.12 -y
rm -rf /ext3/py3.12/compiler_compat/ld
echo "conda activate /ext3/py3.12" >> /ext3/env.sh
source /ext3/env.sh
pip install setuptools==68.2.2 Cython
EOF
${run_desc_script} bash ${tmp_script}


singularity exec --nv --overlay ${base_dir}/desc.ext3 /share/apps/images/ubuntu-24.04.3.sif /bin/bash -c \
"
echo \"source /opt/apps/lmod/lmod/init/sh\" > /ext3/env.sh && \
echo \"export MODULEPATH=/ext3/apps/modulefiles\" >> /ext3/env.sh && \
bash ${base_dir}/setup-conda.bash && \ 
source /ext3/env.sh && \
conda create -p /ext3/py3.12 python=3.12 -y && \
conda activate /ext3/py3.12 && \
git clone https://github.com/masonhaberle/DESC.git && \
cd DESC && \
pip install -e . && \
pip install jupyter jupyterhub jupyterlab pandas matplotlib scipy qsc autopep8 flake8 ruff kaleido && \
plotly_get_chrome && \
rm -rf Miniconda3-*.sh tmp-*.bash && \
echo \"conda activate /ext3/py3.12\" >> /ext3/env.sh
"

