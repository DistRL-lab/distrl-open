import os
import argparse
import subprocess


def execute_adb(adb_command):
    result = subprocess.run(adb_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    print(f"[!] Command execution failed: {adb_command}")
    print(result.stderr)
    return "ERROR"


def get_screenshot(prefix, save_dir=".", screenshot_dir="/sdcard/"):
    cap_command = f"adb shell screencap -p " \
                    f"{os.path.join(screenshot_dir, prefix + '.png')}"
    pull_command = f"adb pull " \
                    f"{os.path.join(screenshot_dir, prefix + '.png')} " \
                    f"{os.path.join(save_dir, prefix + '.png')}"
    result = execute_adb(cap_command)
    if result != "ERROR":
        result = execute_adb(pull_command)
        if result != "ERROR":
            save_path = os.path.join(save_dir, prefix + '.png')
            print(f"screenshot saved to {save_path}")
            return save_path
        return result
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-p', '--prefix', type=str, default="test", required=False)
    args = parser.parse_args()

    get_screenshot(args.prefix)
