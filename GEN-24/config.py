import os
from typing import Dict, Any

def read_api_key(file_path: str) -> str:
    """
    Reads the API key from the specified file.

    Args:
        file_path (str): Path to the file containing the API key.

    Returns:
        str: The API key.

    Raises:
        ValueError: If the API key is not found in the file.
        FileNotFoundError: If the file does not exist.
    """
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith("OPENAI_API_KEY"):
                    return line.split('=')[1].strip()
        raise ValueError("API key not found in the specified file")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} does not exist")

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the LLM_keys.key file
API_KEY_FILE = os.path.join(SCRIPT_DIR, 'LLM_keys.key')

# Ensure output directories exist
def ensure_directories(dirs: Dict[str, str]):
    """
    Ensures that the specified directories exist.

    Args:
        dirs (Dict[str, str]): Dictionary of directory paths.
    """
    for key, path in dirs.items():
        os.makedirs(path, exist_ok=True)

# User-configurable parameters
USER_CONFIG = {
    'output_dirs': {
        'Pro-Introduction_Letter': os.path.join(SCRIPT_DIR, 'Inputs_and_Outputs', 'Pro-Introduction_Letter'),
        'Written_Introduction_Letter': os.path.join(SCRIPT_DIR, 'Inputs_and_Outputs', 'Written_Introduction_Letter')
    },
    'llm_params': {
        'model': 'gpt-4o-mini-2024-07-18',
        'max_tokens': 4096,
        'temperature': 0.5,
        'top_p': 1.0,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
        'n': 1,
        'stream': False,
        'stop': None,
        'api_key': read_api_key(API_KEY_FILE)  # Read the API key from the file
    }
}

# Ensure output directories exist
ensure_directories(USER_CONFIG['output_dirs'])

# Default configuration
DEFAULT_CONFIG = {
    'prompts': {
        'pro_shift': "Generate a pro-shifted domain combining the following domains: {domains}",
        'domain_shift': "Shift the following domain: {domain}",
        'dissertation_outline': "Create a dissertation outline for the following shifted domain: {shifted_domain}",
        'pro_grant': "Prepare a grant proposal for exploring, characterizing, and applying the following shifted domain: {shifted_domain}"
    },
    'llm_params': USER_CONFIG['llm_params'],
    'output_dirs': USER_CONFIG['output_dirs']
}