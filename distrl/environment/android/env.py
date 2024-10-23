import os
import shutil
import subprocess, signal
import re
import shlex
from time import sleep
import random
from .autoui_utils import autoui_prepare_prompt, AndroidAction, ActionType, ImageFeatureExtractor
import time
from distrl.misc import colorful_print

from appium import webdriver
from appium.options.android import UiAutomator2Options

import base64
from PIL import Image
from io import BytesIO
from termcolor import colored, cprint
import concurrent.futures
import numpy as np
import traceback


def escape_shell_text(text):
    # List of characters to escape
    chars_to_escape = ['\\','"', "'", '`', '$']
    
    # Escape the characters by adding a backslash before them
    for char in chars_to_escape:
        text = text.replace(char, '\\' + char)
    text = text.replace(" ", "%s")
    return text


def list_all_devices(adb_path):
    # Get the list of connected devices
    result = subprocess.run([adb_path, 'devices'], stdout=subprocess.PIPE)
    devices_output = result.stdout.decode('utf-8')
    devices_states = re.findall(r'(emulator-\d+)\t(\w+)', devices_output)
    running_emulators = [device_state[0] for device_state in devices_states]
    emulator_status = [device_state[1] for device_state in devices_states]
    return running_emulators, emulator_status


def kill_all_emulators(adb_path, emulators=None):
    # Find all emulator device names using a regular expression
    running_emulators, _ = list_all_devices(adb_path)
    
    # Shut down each emulator found
    for emulator in emulators:
        if emulator not in running_emulators:
            continue
        subprocess.run([adb_path, '-s', emulator, 'emu', 'kill'])
        print(f'{emulator} has been shut down.')

    if not emulators:
        print("No running emulators found.")


def clone_avd(src_avd_name, tar_avd_name, android_avd_home):
    """
    Clone the source AVD to the target AVD.

    Parameters:
    - src_avd_name: The name of the source AVD folder.
    - tar_avd_name: The name of the target AVD folder.
    - android_avd_home: The path to the .android/avd directory.

    This function copies the source AVD folder and its .ini file to a new target AVD
    and updates the paths inside the .ini files accordingly.
    """

    # Paths for source and target AVD directories and .ini files
    src_avd_dir = os.path.join(android_avd_home, src_avd_name + '.avd')
    tar_avd_dir = os.path.join(android_avd_home, tar_avd_name + '.avd')
    src_ini_file = os.path.join(android_avd_home, src_avd_name + '.ini')
    tar_ini_file = os.path.join(android_avd_home, tar_avd_name + '.ini')

    # Copy the AVD folder
    colorful_print(f"Copying the AVD folder from {src_avd_dir} to {tar_avd_dir}", "green")
    if not os.path.exists(tar_avd_dir):
        shutil.copytree(src_avd_dir, tar_avd_dir)

    # Copy the .ini file and modify it for the new AVD
    with open(src_ini_file, 'r') as src_ini, open(tar_ini_file, 'w') as tar_ini:
        for line in src_ini:
            tar_ini.write(line.replace(src_avd_name, tar_avd_name))

    # Update paths inside the target AVD's .ini files
    for ini_name in ['config.ini', 'hardware-qemu.ini']:
        ini_path = os.path.join(tar_avd_dir, ini_name)
        if os.path.exists(ini_path):
            with open(ini_path, 'r') as file:
                lines = file.readlines()
            with open(ini_path, 'w') as file:
                for line in lines:
                    # Update paths and AVD name/ID
                    new_line = line.replace(src_avd_name, tar_avd_name)
                    file.write(new_line)

    # Update the snapshots' hardware.ini file if it exists
    snapshots_hw_ini = os.path.join(tar_avd_dir, 'snapshots', 'default_boot', 'hardware.ini')
    if os.path.exists(snapshots_hw_ini):
        with open(snapshots_hw_ini, 'r') as file:
            lines = file.readlines()
        with open(snapshots_hw_ini, 'w') as file:
            for line in lines:
                # Update AVD name/ID
                new_line = line.replace(src_avd_name, tar_avd_name)
                file.write(new_line)


def execute_adb(adb_command):
    result = subprocess.run(adb_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    print(f"[!] Command execution failed: {adb_command}")
    print(result.stderr)
    return "ERROR"


class EmulatorDriver():
    def __init__(self, udid, adb_path, screenshot_dir="/sdcard"):
        self.udid = udid
        self.adb_path = adb_path
        self.screenshot_dir = screenshot_dir
        assert self._check_device(), f"[!] Cannot connect to emulator {self.udid}"

    def _check_device(self):
        for _ in range(10):
            running_emulators, emulator_status = list_all_devices(self.adb_path)
            if self.udid in running_emulators:
                udid_idx = running_emulators.index(self.udid)
                if emulator_status[udid_idx] == "device":
                    return True
            time.sleep(5)
        return False
        
    def get_window_size(self):
        adb_command = f"adb -s {self.udid} shell wm size"
        result = execute_adb(adb_command)
        if result != "ERROR":
            numbers = re.findall(r'\d+', result)
            if len(numbers) < 2:
                raise ValueError("Expected to find more than two numbers in the adb output.")
            return {"width": int(numbers[0]),
                    "height": int(numbers[1])}
        return {"width": 0,
                "height": 0}

    def get_screenshot(self, prefix, save_dir):
        cap_command = f"adb -s {self.udid} shell screencap -p " \
                      f"{os.path.join(self.screenshot_dir, prefix + '.png')}"
        pull_command = f"adb -s {self.udid} pull " \
                       f"{os.path.join(self.screenshot_dir, prefix + '.png')} " \
                       f"{os.path.join(save_dir, prefix + '.png')}"
        result = execute_adb(cap_command)
        if result != "ERROR":
            result = execute_adb(pull_command)
            if result != "ERROR":
                return os.path.join(save_dir, prefix + '.png')
            return result
        return result

    def tap(self, x, y):
        adb_command = f"adb -s {self.udid} shell input tap {x} {y}"
        ret = execute_adb(adb_command)
        return ret

    def swipe(self, x, y, l_x, l_y):
        duration = 200
        adb_command = f"adb -s {self.udid} shell input swipe {x} {y} {l_x} {l_y} {duration}"
        ret = execute_adb(adb_command)
        return ret

    def text(self, input_str):
        input_str = input_str.replace("'", "")
        input_str = shlex.quote(input_str)
        # input_str = input_str.replace(" ", "%s")
        adb_command = f"adb -s {self.udid} shell input text \"{input_str}\""
        ret = execute_adb(adb_command)
        return ret

    def keyevent(self, event_code):
        adb_command = f"adb -s {self.udid} shell input keyevent {event_code}"
        ret = execute_adb(adb_command)
        return ret


class AndroidEmulator():
    def __init__(self, avd_name, max_steps, temp_path, evaluator, emulator_path="~/Android/Sdk/emulator/emulator", no_window=False, udid = None,
        feature_extractor=None, all_tasks=None, prepare_prompt=autoui_prepare_prompt, translate_action=None, save_images=False, task_id=None, sample_mode=None,
        android_avd_home=None, adb_path="~/Library/Android/sdk/platform-tools/adb"):
        """
        temp_path temporary path to store the images for evaluation
        """
        self.temp_path = temp_path
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        self.emulator_path = os.path.expanduser(emulator_path)
        self.avd_name = avd_name
        self.adb_path = adb_path
        self.save_images = save_images
        self.image_id = str(time.time())
        port_number = udid.split("-")[-1]
        self.udid = udid
        print("UDID", self.udid)

        # setup emulator and driver
        for _ in range(3):
            try:
                cprint(colored(f"Starting the Emulator", "green"))
                command_env_prefix = f"ANDROID_AVD_HOME={android_avd_home} " if android_avd_home else ""
                command = f"""{command_env_prefix}{self.emulator_path} -avd {self.avd_name} "-no-audio" "-skip-adb-auth" "-no-boot-anim" "-gpu" "auto" "-no-snapshot-save" -port {port_number}"""
                if no_window:
                    command += " -no-window"
                command += " -feature -Vulkan"
                print(f"[.] Executing command {command}")
                self.emulator_process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # TODO: check process status
                sleep(15)
                print("[.] Creating drivers...")
                self.driver = EmulatorDriver(self.udid, self.adb_path)
                print("[.] Drivers connected...")
                break
            except Exception as err:
                print(f"Start emulator {self.udid} failed: {err}, retrying...")
                self.terminate()

        self.terminated = False
        self.max_steps = max_steps
        self.steps = 0
        self.feature_extractor = feature_extractor
        screen_size = self.driver.get_window_size()
        print(f"[.] Get window size {screen_size}")
        self.screen_size = (screen_size["width"], screen_size["height"])
        if sample_mode == "random":
            # randomly sample a task from the task set
            self.current_task = random.choice(all_tasks)
        elif sample_mode == "sequential":
            self.current_task = all_tasks[task_id]
            print(f"Current task: {task_id}")
        else:
            print("Invalid sample mode")
        self.prepare_prompt = prepare_prompt
        self.translate_action = translate_action
        self.history = []
        self.evaluator = evaluator

        self.last_valid_action = None
        self.last_repetition_penalty = 0
    
    def terminate(self):
        # sleep(5)
        self.emulator_process.terminate()
        try:
            try:
                self.emulator_process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.emulator_process.kill()
                self.emulator_process.wait()
        except Exception as err:
            print(f"[!] Exception during kill emulator process: {err}")
        
        # double check if the emulator has been stopped
        try:
            subprocess.run([self.adb_path, '-s', self.udid, 'emu', 'kill'])
            print(f'{self.udid} has not stopped yet and has been shut down.')
        except Exception:
            print(f'{self.udid} has been shut down.')
        self.terminated = True
    
    def count_white_pixels(self, img):
        # Convert the image to RGB format if it's not
        img = img.convert('RGB')
        # Convert image to numpy array
        data = np.array(img)
        # Count white pixels
        # Assuming 'white' is (255, 255, 255)
        white_count = np.sum(np.all(data > 240, axis=-1))
        return white_count > 2_300_000
    
    def get_obs(self):
        for _ in range(3):
            try:
                is_white = True
                for _ in range(5):
                    if not is_white:
                        break
                    sleep(5)
                    screenshot_path = self.driver.get_screenshot(prefix=f"{self.image_id}_{self.steps}", save_dir=self.temp_path)
                    # screenshot_str = self.driver.get_screenshot_as_base64()
                    # imgdata = base64.b64decode(screenshot_str)
                    # image =  Image.open(BytesIO(imgdata))
                    image = Image.open(screenshot_path)
                    is_white = self.count_white_pixels(image)
                # print("Saving observation!")
                # image.save(os.path.join(self.temp_path, f"{self.image_id}_{self.steps}.png"))
                # Assuming 'image' is your PIL Image object in RGBA mode
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                if self.feature_extractor is not None:
                    image = self.feature_extractor.to_feat(image)
                # colorful_print(f"history: {self.history}", "green")
                # colorful_print(f"prompt: {self.prepare_prompt(self.current_task, self.history)}", "yellow")
                return {"prompt": self.prepare_prompt(self.current_task, self.history),
                        "image_feature": image,
                        "task": self.current_task,
                        "image_path": os.path.join(self.temp_path, f"{self.image_id}_{self.steps}.png")
                }
            except Exception as e:
                print(f"Exception happened during screenshotting")
                print(e)
                print(traceback.format_exc())
                sleep(6)
                continue
    
    def step(self, raw_action: str):
        if self.terminated:
            return None
        try:
            # colorful_print(f"raw action: {raw_action}", "green")
            action = self.translate_action(raw_action)
            # colorful_print(f"translated action: {action}", "green")
        except Exception as e:
            print(e)
            print(f"Failed to translate action: {raw_action}, terminating the environment")
            action = AndroidAction(action_type=ActionType.TaskImpossible)
        self.history.append(action)
        self.steps += 1
        if self.steps > self.max_steps:
            action = AndroidAction(action_type=ActionType.TaskImpossible)
            cprint(colored(f"Terminate the Emulator: Max Steps Exceeded {self.max_steps}.", "red"))
        screenshot = None
        info = {}
        for i in range(2):
            try:
                if action.action_type == ActionType.DualPoint:
                    assert len(action.touch_point) == 2
                    assert len(action.lift_point) == 2
                    touch_x = action.touch_point[0] * self.screen_size[0]
                    touch_y = action.touch_point[1] * self.screen_size[1]
                    lift_x = action.lift_point[0] * self.screen_size[0]
                    lift_y = action.lift_point[1] * self.screen_size[1]
                    if (touch_x - lift_x)**2 + (touch_y - lift_y)**2 < 10:
                        self.driver.tap(touch_x, touch_y)
                    else:
                        self.driver.swipe(touch_x, touch_y, lift_x, lift_y)
                elif action.action_type == ActionType.Type:
                    action.typed_text = action.typed_text.strip()
                    self.driver.text(action.typed_text)
                elif action.action_type == ActionType.GoBack:
                    self.driver.keyevent(4)
                elif action.action_type == ActionType.GoHome:
                    self.driver.keyevent(3)
                elif action.action_type == ActionType.Enter:
                    self.driver.keyevent(66)
                elif action.action_type == ActionType.TaskComplete:
                    self.terminated = True
                elif action.action_type == ActionType.TaskImpossible:
                    self.terminated = True
                elif action.action_type == ActionType.Idle:
                    pass
                else:
                    raise Exception(f"Unknown action type: {action.action_type}")
                action_success = True
                screenshot = self.get_obs()
                break
            except Exception as e:
                cprint(colored("an Exception occurred during environment interaction", "red"))
                print(e)
                cprint(colored("Retrying", "red"))
                sleep(10)
                if i == 1:
                    action_success = False
                    info["error"] = str(e)
                    self.terminate()
                    return None
                continue
        r = 0
        penalty = 0
        action_valid = True
        get_repetition = False
        # check invalid action / repetition
        if action.action_type == ActionType.Type and not action.typed_text.strip():
            penalty = -1
            action_valid = False
        elif self.last_valid_action and self.last_valid_action.__str__() == action.__str__():
            penalty = self.last_repetition_penalty - 0.4
            self.last_repetition_penalty = penalty
            get_repetition = True

        if screenshot is not None and self.evaluator is not None:
            print("Evaluating...")
            r = self.evaluator([os.path.join(self.temp_path, f"{self.image_id}_{self.steps-1}.png"), 
                                os.path.join(self.temp_path, f"{self.image_id}_{self.steps}.png")], self.current_task)
            if r == -1:
                r = 0
                penalty = -1
                action_valid = False
        
        if action_valid:
            self.last_valid_action = action
        if not get_repetition:
            self.last_repetition_penalty = 0

        info["action_success"] = action_success
        #terminate the environment if there is a success
        if r >= 1 or self.terminated:
            self.terminate()
            print("Terminating emulator and driver...")
        if self.terminated and not self.save_images:
            os.system(f"rm -rf {self.temp_path}/*")
            print("Deleting saved screenshots...")
        return screenshot, r, self.terminated, penalty


class BatchedAndroidEnv():
    """
    This class wraps around the android emulator and provides a more infrastructure for free-form GUI navigation
    This is a batched version for Android Env
    cache_avd is the avd to be used the avd is the initial one
    """
    def __init__(self, 
        avd_name, 
        cache_avd_names,
        udids,
        android_avd_homes,
        emulator_path: str = '~/Android/Sdk/emulator/emulator',
        adb_path: str = "~/Library/Android/sdk/platform-tools/adb",
        run_headless: bool = False,
        max_steps: int = 10,
        use_feature_extractor = False, 
        evaluators = None,
        prepare_prompt = autoui_prepare_prompt, 
        translate_action = None,
        device = "cuda:2",
        temp_path = "~/logs/worker/images",
        save_images = False,
        all_tasks = None,
        sample_mode = None
    ):
        
        self.android_avd_homes = [os.path.expanduser(android_avd_home) for android_avd_home in android_avd_homes]
        self.emulator_path = os.path.expanduser(emulator_path)
        self.adb_path = os.path.expanduser(adb_path)
        self.avd_name = avd_name
        self.save_images = save_images
        self.bsize = len(cache_avd_names)
        self.cache_avd_names = cache_avd_names
        self.run_headless = run_headless
        self.max_steps = max_steps
        self.emulator_group_offset = 0
        if use_feature_extractor:
            self.feature_extractor = ImageFeatureExtractor("cpu")
        else:
            self.feature_extractor = None
        self.device = device
        self.all_tasks = all_tasks
        self.prepare_prompt = prepare_prompt
        self.translate_action = translate_action
        self.temp_path = temp_path
        if evaluators is None:
            evaluators = [None for _ in range(self.bsize)]
        self.evaluators = evaluators
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        self.udids = udids
        self.sample_mode = sample_mode

    def reset(self):
        """
        Reset the emulator to a clean state
        """
        # If the emulator is already running, kill it,
        # Then delete the cache AVD
        kill_all_emulators(self.adb_path, emulators=self.udids)
        if hasattr(self, "emulator_process"):
            self.emulator_process.send_signal(signal.SIGINT)
            self.emulator_process.wait()
        self.emulators = []

        # clond avd
        # for cache_avd_name, android_avd_home in zip(self.cache_avd_names, self.android_avd_homes):
        #     # print(cache_avd_name)
        #     for _ in range(3):
        #         try:
        #             cache_avd_path = os.path.join(android_avd_home, cache_avd_name + ".avd")
        #             cache_avd_ini_path = os.path.join(android_avd_home, cache_avd_name + ".ini")
        #             if os.path.exists(cache_avd_path):
        #                 shutil.rmtree(cache_avd_path, ignore_errors=True)
        #             if os.path.exists(cache_avd_ini_path):
        #                 os.remove(cache_avd_ini_path)
        #             sleep(2)
        #             # Clone the source AVD and start the emulator
        #             clone_avd(self.avd_name, cache_avd_name, android_avd_home)
        #             break
        #         except OSError as e:
        #             print(f"Failed to reset the emulator: {e}")
        #             import traceback
        #             print(traceback.format_exc())
        #             sleep(20)
        
        # use parallel version only when you've got nice CPUs, or it will error out
        job_group_list = []
        for cache_avd_name, android_avd_home in zip(self.cache_avd_names, self.android_avd_homes):
            finish = False
            for group in job_group_list:
                if android_avd_home in group.keys():
                    continue
                else:
                    group[android_avd_home] = cache_avd_name
                    finish = True
            if not finish:
                job_group_list.append({android_avd_home: cache_avd_name})
        
        def reset_emulator(cache_avd_name, avd_name, android_avd_home):
            for _ in range(3):
                try:
                    cache_avd_path = os.path.join(android_avd_home, cache_avd_name + ".avd")
                    cache_avd_ini_path = os.path.join(android_avd_home, cache_avd_name + ".ini")
                    if os.path.exists(cache_avd_path):
                        shutil.rmtree(cache_avd_path, ignore_errors=True)
                    if os.path.exists(cache_avd_ini_path):
                        os.remove(cache_avd_ini_path)
                    sleep(2)
                    # Clone the source AVD and start the emulator
                    clone_avd(avd_name, cache_avd_name, android_avd_home)
                    break
                except OSError as e:
                    print(f"Failed to reset the emulator: {e}")
                    import traceback
                    print(traceback.format_exc())
                    sleep(20)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for group in job_group_list:
                futures = [executor.submit(reset_emulator, cache_avd_name, self.avd_name, android_avd_home) for android_avd_home, cache_avd_name in group.items()]
                for future in futures:
                    future.result()

        def emulator_constructor(udid, cache_avd_name, android_avd_home, evaluator, task_id):
            return AndroidEmulator(avd_name=cache_avd_name, max_steps=self.max_steps, emulator_path=self.emulator_path, 
                no_window=self.run_headless, 
                udid = udid,
                feature_extractor = self.feature_extractor,
                prepare_prompt = self.prepare_prompt,
                translate_action = self.translate_action,
                all_tasks = self.all_tasks,
                evaluator = evaluator,
                temp_path = os.path.join(self.temp_path, cache_avd_name),
                save_images = self.save_images,
                task_id=task_id,
                sample_mode=self.sample_mode,
                android_avd_home=android_avd_home,
                adb_path=self.adb_path)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(emulator_constructor, udid, cache_avd_name, android_avd_home, evaluator, task_id)
                for udid, cache_avd_name, android_avd_home, evaluator, task_id in 
                zip(self.udids, self.cache_avd_names, self.android_avd_homes, self.evaluators, range(self.emulator_group_offset, self.emulator_group_offset+self.bsize))]
            self.emulators = [job.result() for job in jobs]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(emulator.get_obs) for emulator in self.emulators]
            # for i, job in enumerate(jobs):
                # colorful_print(f"Getting observation from emulator {i}: {job.result()}", "green")
            return [job.result() for job in jobs]

    def step(self, actions):
        if not self.emulators:
            raise Exception("Please call reset() before calling step()")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(emulator.step, action) 
                    for emulator, action in 
                    zip(self.emulators, actions)]
            results = [job.result() for job in jobs]
        return results
