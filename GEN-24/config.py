import os
from typing import Dict

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the LLM_keys.key file (read lazily by utils.get_api_key()).
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
        'model': 'gpt-4o-mini',
        'max_tokens': 4096,
        'temperature': 0.5,
        'top_p': 1.0,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
        'n': 1,
        'stream': False,
        'stop': None,
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