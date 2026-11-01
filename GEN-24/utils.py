from config import DEFAULT_CONFIG
import os
import logging
import time
from typing import Any, Dict, Tuple
from openai import OpenAI

client = None

def initialize_openai_client(api_key=None):
    global client
    if api_key is None:
        api_key = get_api_key()
    client = OpenAI(api_key=api_key)
    logging.info("OpenAI client initialized successfully")

def load_file(file_path: str) -> str:
    """
    Load the contents of a file.

    Args:
        file_path (str): Path to the file to be loaded.

    Returns:
        str: Contents of the file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: If there's an error reading the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        logging.info(f"File loaded successfully: {file_path}")
        return content
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except IOError as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")
        raise

def save_file(content: str, file_path: str) -> None:
    """
    Save content to a file.

    Args:
        content (str): Content to be saved.
        file_path (str): Path where the file should be saved.

    Raises:
        IOError: If there's an error writing to the file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        logging.info(f"File saved successfully: {file_path}")
    except IOError as e:
        logging.error(f"Error writing to file {file_path}: {str(e)}")
        raise

def generate_llm_response(prompt: str, llm_params: Dict[str, Any] = DEFAULT_CONFIG['llm_params']) -> Tuple[str, float]:
    """
    Generate a response from the LLM.

    Args:
        prompt (str): The input prompt for the LLM.
        llm_params (Dict[str, Any]): LLM parameters from config.

    Returns:
        Tuple[str, float]: The generated response from the LLM and the time taken in seconds.
    """
    global client
    if client is None:
        initialize_openai_client()

    start_time = time.time()
    try:
        response = client.chat.completions.create(model=llm_params.get('model'),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=llm_params.get('max_tokens'),
        temperature=llm_params.get('temperature'),
        top_p=llm_params.get('top_p'),
        frequency_penalty=llm_params.get('frequency_penalty'),
        presence_penalty=llm_params.get('presence_penalty'),
        n=llm_params.get('n'),
        stream=llm_params.get('stream'),
        stop=llm_params.get('stop'))
        generated_response = response.choices[0].message.content.strip()
        logging.info("LLM response generated successfully")
    except Exception as e:
        logging.error(f"Error generating LLM response: {str(e)}")
        raise
    finally:
        end_time = time.time()

    time_taken = end_time - start_time
    logging.info(f"LLM response generated in {time_taken:.2f} seconds")
    return generated_response, time_taken

def get_api_key() -> str:
    """
    Read the OpenAI API key from the LLM_keys.key file.

    Returns:
        str: The OpenAI API key.

    Raises:
        ValueError: If the API key is not found in the file.
    """
    key_file_path = os.path.join(os.path.dirname(__file__), 'LLM_keys.key')
    try:
        with open(key_file_path, 'r') as key_file:
            for line in key_file:
                if line.startswith('OPENAI_API_KEY'):
                    api_key = line.split('=')[1].strip()
                    logging.info("API key read successfully")
                    return api_key
    except FileNotFoundError:
        logging.error(f"API key file not found: {key_file_path}")
    except Exception as e:
        logging.error(f"Error reading API key: {str(e)}")

    raise ValueError("OPENAI_API_KEY not found in LLM_keys.key file")