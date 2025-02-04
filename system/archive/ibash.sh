cd "$(dirname "$0")"
cd ../../infra/setup 
source utils.sh


add_line_to_bashrc "alias suffer='trap \"pkill -P \$\$\" SIGINT; stress --cpu $(nproc) --timeout 300 & gpu_burn 1000 &'"
source ~/.bashrc