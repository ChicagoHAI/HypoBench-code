"""
This model initializes language models, and provide a comprehensive interface. 
"""

from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import os
import weave
import json
from typing import List, Dict, Tuple, Union

format_prompt_str = """Consider the given response below: 
{response}

Your objective is to format the response according to the provided response_format. 
"""

class AbstractLanguageModel:
    def __init__(self, **kwargs):
        pass
    
    def _generate(self, **kwargs):
        pass

    def batch_generate(self, **kwargs):
        pass


class GPT4(AbstractLanguageModel):
    def __init__(self, max_tokens=5000, temperature=0.1, top_p=0.9, max_concurrent=10, 
                 model_name="gpt-4o-mini", **kwargs):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_concurrent = max_concurrent
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.token_usage = {"prompt": 0, "response": 0, "prompt_cached": 0}

    @weave.op()
    def _generate(self, prompt, response_format=None, temperature=0.0, top_p=0.9, **kwargs):
        """
        prompt: a list of dictionaries, each dictionary contains the role and content of the message.

        Generate the response based on the prompt. 
        """
        if "o1" in self.model_name:
            """
            o1 models don't support system message, therefore need to integrate system messages into user messages. 
            """
            if len(prompt) > 1: # system prompt exists
                prompt = [{"role": "user", "content": "\n\n".join([f"{msg['content']}" for msg in prompt])}]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
            )
            self.token_usage["prompt"] += response.usage.prompt_tokens
            self.token_usage["response"] += response.usage.completion_tokens
            content_str = response.choices[0].message.content
            # ask GPT-4o-mini to format according to the response_format
            if response_format is not None:
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": format_prompt_str.format(response=content_str)}],
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    response_format=response_format,
                )
                self.token_usage["prompt"] += response.usage.prompt_tokens
                self.token_usage["response"] += response.usage.completion_tokens
                self.token_usage["prompt_cached"] += response.usage.prompt_tokens_details["cached_tokens"]
                return response.choices[0].message.content  # whether the response is a json or not will be handled at where this method is called.
            else:
                return content_str
        else: 
            attempts = 5
            while attempts > 0:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=prompt,
                    max_tokens=self.max_tokens,
                    temperature=temperature, 
                    top_p=self.top_p,
                    response_format=response_format, 
                    # logprobs=True, 
                ) 
                self.token_usage["prompt"] += response.usage.prompt_tokens
                self.token_usage["response"] += response.usage.completion_tokens
                self.token_usage["prompt_cached"] += response.usage.prompt_tokens_details.cached_tokens
                if response_format is None:
                    return response.choices[0].message.content  # whether the response is a json or not will be handled at where this method is called.
                else: # apply json loading exception handling to try avoiding exceptions due to erroneous json. 
                    try:
                        json.loads(response.choices[0].message.content)  # check if the response is a valid json
                        return response.choices[0].message.content
                    except Exception as e:
                        print(f"Error: {e}")
                        attempts -= 1
                        temperature = min(temperature + 0.1, 1.0)

            return response.choices[0].message.content  # whether the response is a json or not will be handled at where this method is called. 

    @weave.op()
    def batch_generate(self, prompt_list: List[List[Dict[str, str]]], response_format=None, **kwargs):
        """
        prompt_list: a list of prompts, each prompt is a list of dictionaries, each dictionary contains the role and content of the message.

        Generate the response based on the prompt. 
        """
        if kwargs.get("temperature") is not None:
            temperature = kwargs["temperature"]
        else:
            temperature = self.temperature
        if kwargs.get("top_p") is not None:
            top_p = kwargs["top_p"]
        else:
            top_p = self.top_p
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            responses = list(executor.map(lambda prompt: self._generate(prompt, response_format=response_format, temperature=temperature, top_p=top_p), prompt_list))
        return responses # each is a list of string responses. 
    

        
    def get_token_usage(self):
        """
        Get the token usage. 
        """
        return self.token_usage
    
    def reset_token_usage(self):
        """
        Reset the token usage. 
        """
        self.token_usage = {"prompt": 0, "response": 0, "prompt_cached": 0}

@weave.op()
def self_consistency_prompting(model, prompt_list: List[List[Dict[str, str]]], response_format, result_key, num_samples=9, temperature=1.0, top_p=0.5, **kwargs):
    assert len(prompt_list) == 1, "The prompt_list should contain only one prompt."
    prompts = prompt_list * num_samples
    with ThreadPoolExecutor(max_workers=3) as executor:
        responses = list(executor.map(lambda prompt: model.batch_generate([prompt], response_format=response_format, temperature=temperature, top_p=top_p), prompts))
    all_responses = [json.loads(res[0]) for res in responses]
    all_res = [json.loads(res[0])[result_key] for res in responses]
    # calculate the percentage of each option's occurrence
    percentage = {key: all_res.count(key) / num_samples for key in set(all_res)}
    
    return max(set(all_res), key=all_res.count), percentage, all_responses


# debug self_consistency_prompting: 
'''
model = GPT4()
prompt = [{"role": "system", "content": "This is a system message."}, {"role": "user", "content": "Test question: Which food would you prefer? A) Pizza B) Burger"}]
feature_generation_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "continuation",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "enum": ["A", "B"]
                },
            },
            "required": ["answer"],
            "additionalProperties": False
        },
        "strict": True
    }
}
self_consistency_prompting(model, [prompt], feature_generation_response_format, "answer", num_samples=9, temperature=1.0, top_p=0.5)
'''