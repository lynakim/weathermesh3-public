import argparse
import os
from neovis import Neovis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str, help='Name of the run directory in /huge/deep/runs/, e.g. run_Sep24-YB-serp3rand144hr-rtft_20240924-181637')
    args = parser.parse_args()

    outpath = '/fast/to_srv/train_imgs/'
    run_name = args.run_name
    path = "/huge/deep/runs/" + run_name
    assert os.path.exists(path), f"Run {run_name} does not exist"

    os.makedirs(outpath, exist_ok=True)
    os.system(f'ln -sn {path}/imgs {outpath}/{run_name}')

    n = Neovis(path + '/imgs')
    n.make(anti_alias=False)
    print(f"Training images for run {run_name} are now available at https://a.windbornesystems.com/indexed/asdf/fast/train_imgs/{run_name}/view.html")

if __name__ == '__main__':
    main()
