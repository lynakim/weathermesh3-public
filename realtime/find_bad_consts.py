import datetime
import os
import numpy as np
import sys

CONSTS_DIR = "/fast/consts"

REMOVE = '--remove' in sys.argv
CHECK_RADIATION_1 = '--no-radiation-1' not in sys.argv
CHECK_NEORADIATION_1 = '--no-neoradiation-1' not in sys.argv
CHECK_SOLARANGLE_1 = '--no-solarangle-1' not in sys.argv


def check_file(file):
    if not os.path.exists(file):
        print(f"{file} does not exist")
        return False

    try:
        np.load(file, allow_pickle=True)
    except Exception as e:
        print(f"{file} is bad: {e}")

        if REMOVE:
            os.remove(file)
            print(f"Removed {file}")

        return False


def check_day(day):
    if CHECK_RADIATION_1:
        check_file(f"{CONSTS_DIR}/radiation_1/{day}.npy")

    for hour in range(24):
        if CHECK_NEORADIATION_1:
            check_file(f"{CONSTS_DIR}/neoradiation_1/{day}_{hour}.npy")

        if CHECK_SOLARANGLE_1:
            check_file(f"{CONSTS_DIR}/solarangle_1/{day}_{hour}.npy")


def main():
    # start at today, go around in a circle
    start_day = datetime.datetime.now().timetuple().tm_yday
    for day_offset in range(0, 366):
        current_day = ((start_day + day_offset) % 365) + 1
        check_day(current_day)


if __name__ == "__main__":
    main()
