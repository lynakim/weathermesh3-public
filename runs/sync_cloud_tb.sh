sync_tb() {
    rsync -av --progress --include='*events.out.tfevents*' --exclude='*' $2:/fast/deep/runs/$1/ /huge/deep/runs/$1/ 
}

sync_tb $1 ${2:-"nimbus"} 
#eg. ./sync_tb_aws run_Sep26-clouddoctor_20240907-012345 cirrus


