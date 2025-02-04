from setuptools import setup

setup(
    name='dlnwp_tools',
    version='0.1',
    packages=[''],
    package_dir={'': '.'},
    entry_points={
        'console_scripts': [
            'lsg = dltools:ls_gpu',
            'lsr = dltools:ls_runs',
            'lsga = dltools:ls_gpu_avg',
            'lss = dltools:ls_screens',
        ],
    },
)