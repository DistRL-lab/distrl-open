import torch
import requests
import re
from typing import List, Tuple
from tenacity import retry, wait_chain, wait_fixed, stop_after_attempt

class ActionValidityChecker:
    def __init__(self, gemini_key=None):
        self.model_name = "gemini-1.5-pro-latest"
        self.api_key = gemini_key

    def check_action_validity(self, actions: List[str]):
        '''
        Checks the validity of a list of actions using Gemini-1.5-pro.

        Args:
            actions: list of actions (e.g., strings)

        Returns:
            torch.Tensor: tensor of penalty values (1 for invalid actions, 0 for valid actions)
        '''
        penalty_values = []
        for action in actions:
            try:
                is_valid = self._evaluate_action(action)
            except Exception as e:
                print(f"Error evaluating action '{action}': {e}")
                is_valid = False  # Default to invalid if there's an error
            penalty = 0 if is_valid else 1
            penalty_values.append(penalty)
        return torch.Tensor(penalty_values)

    def _evaluate_action(self, action: str):
        '''
        Evaluates the validity of a single action using Gemini.

        Args:
            action: The action to be validated.

        Returns:
            bool: True if the action is valid, False otherwise.
        '''
        # Build the prompt with examples
        system_msg, prompt = self.build_prompt(action)

        # Call Gemini API
        response_text = self.call_gemini_url(system_msg, prompt)

        # Extract the status from the response
        status = self.extract_status(response_text)
        if status is not None and 'valid' in status.lower():
            return True
        else:
            return False

    def build_prompt(self, action: str):
        '''
        Builds the prompt for Gemini, including examples.

        Args:
            action: The action to be validated.

        Returns:
            Tuple[str, List[str]]: system message and list of messages for the prompt.
        '''
        system_msg = "You're an expert in evaluating whether the Action is valid based on task-specific criteria."
        prompt = [
            """Action: Open the settings app
                Is this action valid?
                Answer: Valid""",
                            """Action: Fly to the moon without any equipment
                Is this action valid?
                Answer: Invalid""",
                            """Action: Search for "laptop" on bestbuy.com
                Is this action valid?
                Answer: Valid""",
                            """Action: Divide a number by zero
                Is this action valid?
                Answer: Invalid""",
                            f"""Action: {action}
                Is this action valid?
                Answer:"""
        ]
        return system_msg, prompt

    @retry(wait=wait_chain(*[wait_fixed(1) for _ in range(3)] + [wait_fixed(3) for _ in range(2)] + [wait_fixed(5)]),
           stop=stop_after_attempt(5))
    def call_gemini_url(self, system_msg: str, prompt: List[str]):
        '''
        Calls the Gemini API with the given system message and prompt.

        Args:
            system_msg: The system message for Gemini.
            prompt: The list of messages for the prompt.

        Returns:
            str: The response text from Gemini.
        '''
        headers = {"Content-Type": "application/json"}
        input_msg = [{"text": system_msg + "\n=====Examples====="}]
        # Add example prompts
        for p in prompt[:-1]:
            input_msg.append({"text": p})
        # Separator between examples and the actual action
        input_msg.append({"text": "=====Your Turn====="})
        # Add the action to be evaluated
        input_msg.append({"text": prompt[-1]})
        payload = {"instances": [{
            "messages": input_msg
        }]}
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta2/models/{self.model_name}:generateMessage?key={self.api_key}", 
            headers=headers, json=payload
        )
        response.raise_for_status()
        response_data = response.json()
        # Extract the response text
        response_text = response_data['candidates'][0]['content']
        return response_text

    def extract_status(self, text: str):
        '''
        Extracts the status ('Valid' or 'Invalid') from the response text.

        Args:
            text: The response text from Gemini.

        Returns:
            str: 'Valid' or 'Invalid' if found, else None.
        '''
        match = re.search(r'(Valid|Invalid)', text, re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            return None
