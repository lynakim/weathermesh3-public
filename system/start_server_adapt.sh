create_screen() {
  local name=$1
  local cmd=$2
  local log_dir=~/screenlogs

  mkdir -p $log_dir
  if ! screen -list | grep -q "$name"; then
    screen -L -Logfile $log_dir/$name -S $name -dm bash -i -c "$cmd; exec bash"
  fi
}

create_screen "tb-adapt" "tensorboard --logdir=/fast/ignored/runs_adapt/ --bind_all --port 6007"

# tensorboard credentials are
# windborne / vinoddontsurf
