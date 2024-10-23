import json
import requests
from PIL import Image
from typing import List, Tuple
from gradio_client import Client
from transformers import AutoTokenizer
import numpy as np
from gradio_client.utils import QueueError, file
from time import sleep
import re
import os
import io
from tenacity import retry, wait_random_exponential, stop_after_attempt, wait_fixed, wait_chain
import base64
import traceback
import google.generativeai as genai
import subprocess
import time
import signal

def extract_status(text):
    match = re.search(r'Status:\s*(\w+)', text)
    if match:
        return match.group(1)
    else:
        return None



def build_prompt_webshop(intent) -> Tuple[str, List[str], List[str]]:
    system_msg = """You're an expert in evaluating whether the Screenshot successfully completes the Task."""
    prompt = [
        """Task: Go to bestbuy.com
Q: What should I expect to see on the screenshot if I've gone to bestbuy.com?
A: I should expect to see I'm on the Best Buy website, which usually shows the Best Buy logo with some featured products and categories. The screenshot shows I'm searching for "bestbuy.com" in Google search (with some search suggestions) instead of being on the Best Buy website.
Status: failure""",
        # ... Include other examples ...
        f"""Task: {intent}
Respond in this format:
Q: What should I expect to see on the screenshot if I've {intent}?
A: I should expect to see <describe expected outcome>. The screenshot shows <describe actual screenshot>.
Status: success or failure (don't return anything else).
Start with "Q:"."""
    ]
    
    image_paths = os.path.join(os.path.dirname(__file__), "assets", "images")
    cot_image_list = [
        os.path.join(image_paths, "step1_bestbuy.png"),  # Example 0
        # ... Include images for other examples ...
        ""  # Placeholder for current image
    ]
    
    return system_msg, prompt, cot_image_list

def build_prompt_general(intent) -> Tuple[str, List[str], List[str]]:
    # Similar to build_prompt_webshop, include examples and images
    pass

def build_prompt_app_install(intent) -> Tuple[str, List[str], List[str]]:
    # Similar to build_prompt_webshop, include examples and images
    pass

@retry(wait=wait_chain(*[wait_fixed(1) for i in range(3)] + [wait_fixed(3) for i in range(2)] + [wait_fixed(5)]),
         stop=stop_after_attempt(5))
def call_gemini_url(model_name, api_key, system_msg, prompt, image_list, image_path):
    headers = {"Content-Type": "application/json"}
    input_msg = [{"text": system_msg + "\n" + "=====Examples====="}]
    for i in range(len(prompt) - 1):
        input_msg += [
            {"text": "\nScreenshot:"},
            {"inline_data": {
                "mime_type": "image/png",
                "data": encode_image(image_list[i])
            }},
            {"text": prompt[i]}
        ]
    input_msg += [
        {"text": "=====Your Turn====="},
        {"text": "\nScreenshot: "},
        {"inline_data": {
            "mime_type": "image/png",
            "data": encode_image(image_path)
        }},
        {"text": prompt[-1]}
    ]
    payload = {"contents": [{"parts": input_msg}]}
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}",
        headers=headers, json=payload
    )
    response = response.json()
    response_text = response['candidates'][0]['content']['parts'][0]['text']
    return response_text



def process_image(image_path):
    image = Image.open(image_path, 'r')
    image = image.resize((image.width // 4, image.height // 4))
    # Save to a BytesIO object (in-memory file) as PNG
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    # Load it back from the BytesIO object
    buffer.seek(0)
    image_reloaded = Image.open(buffer)
    return image_reloaded


def encode_image(image_path):
    image = Image.open(image_path, 'r')
    image = image.resize((image.width // 4, image.height // 4))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str


class EndResultEvaluator:
    def __init__(self, gemini_key=None, task_set=None):
        genai.configure(api_key=gemini_key)
        self.client = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        self.model_name = "gemini-1.5-pro-latest"
        self.api_key = gemini_key
        self.task_set = task_set

        self.img_matrix = None
        self.cache_max = 5
        self.threshold = 0.001 * 255 ** 2  # Adjust as needed

    def __call__(self, image_path: str, intent: str) -> bool:
        """
        image_path: path to the image to evaluate
        intent: a string representing the user's intent

        Returns:
        - True if the task is completed
        - False otherwise

        If there's an error, it will return False and print the error message
        """
        try:
            with Image.open(image_path) as img_src:
                img = np.array(img_src)
        except Exception as e:
            print(f"Error opening image: {e}")
            return False

        # Check if the image is similar to any previously seen images
        if self.img_matrix is None:
            self.img_matrix = np.expand_dims(img, axis=0)
        else:
            distances = np.mean((self.img_matrix.astype(np.float64) - img.astype(np.float64)) ** 2, axis=(1, 2, 3))
            if np.min(distances) < self.threshold:
                print("Skipping evaluation due to previously seen image, current img_matrix size:", self.img_matrix.shape[0])
                del img
                return False  # or return -1 if you prefer to indicate skipped evaluation
            elif self.img_matrix.shape[0] < self.cache_max:
                self.img_matrix = np.concatenate([self.img_matrix, np.expand_dims(img, axis=0)], axis=0)

        print(f"Task: {intent}, image: {image_path}")
        eval_res = self._evaluate(intent, image_path)

        del img
        return eval_res


    def _evaluate(self, intent: str, image_path: str) -> bool:
        if self.task_set == "general":
            system_msg, prompt, cot_image_list = build_prompt_general(intent)
        elif self.task_set == "webshop":
            system_msg, prompt, cot_image_list = build_prompt_webshop(intent)
        
        print("Calling Gemini...")
        response_text = call_gemini_url(self.model_name, self.api_key, system_msg, prompt, cot_image_list, image_path)
        
        if extract_status(response_text) is not None and 'success' in extract_status(response_text).lower():
            print("Success!")
            print("image path:" + image_path)
            print("prompt")
            print(prompt[-1])
            print("response")
            print(response_text)
            return True
        return False
        