import logging
import os
import time
import openai
from utils import load_file, save_file, generate_llm_response, get_api_key
from config import USER_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the input file and output directory
base_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(base_dir, "Inputs_and_Outputs", "Pro-Introduction_Letter", "introduction_letter_Hipolito_and_Friston.md")
OUTPUT_DIR = os.path.join(base_dir, "Inputs_and_Outputs", "Written_Introduction_Letter")

print("Current working directory:", os.getcwd())

def process_introduction_letter(input_file: str, output_dir: str, llm_params: dict) -> None:
    """
    Process an introduction letter file, send it to LLM, and save the response as markdown.

    Args:
        input_file (str): Path to the input introduction letter file.
        output_dir (str): Directory to save the processed letter.
        llm_params (dict): Parameters for the language model.
    """
    try:
        logger.info(f"Processing file: {input_file}")

        # Load letter content
        letter_content = load_file(input_file)

        # Generate the processed letter using LLM
        processed_letter, _ = generate_llm_response(letter_content, llm_params)

        # Prepare the output file path
        output_file_name = 'written_' + os.path.splitext(os.path.basename(input_file))[0] + '.md'
        output_file_path = os.path.join(output_dir, output_file_name)

        # Save the processed letter as markdown
        save_file(processed_letter, output_file_path)

        logger.info(f"Processed letter saved as markdown: {output_file_name}")
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {str(e)}")

def process_introduction_letters(input_file: str, output_dir: str, config: dict) -> None:
    """
    Process the introduction letter using the provided configuration.

    Args:
        input_file (str): Path to the input introduction letter file.
        output_dir (str): Directory to save the processed letter.
        config (dict): Configuration dictionary.
    """
    try:
        logger.info("Starting introduction letter processing")

        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            # Handle the error or exit the script
            return

        # Log file permissions
        logger.info(f"File permissions for {input_file}: {oct(os.stat(input_file).st_mode)}")

        # Get the API key using the new get_api_key() function
        api_key = get_api_key()
        if not api_key:
            raise ValueError("API key not found in .env or LLM_keys.key file")

        # Prepare LLM parameters
        llm_params = {**USER_CONFIG['llm_params'], 'api_key': api_key}

        # Ensure the output directory exists and is writable
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"Permission denied: '{output_dir}'")

        # Prepare the output file name
        output_file_name = 'written_' + os.path.basename(input_file)
        output_file_path = os.path.join(output_dir, output_file_name)

        # Check if the output file already exists
        if os.path.exists(output_file_path):
            logger.info(f"Output file already exists: {output_file_name}. Skipping LLM request.")
            print(f"Output file already exists: {output_file_name}. Skipping LLM request.")
        else:
            # Process the introduction letter
            start_time = time.time()
            process_introduction_letter(input_file, output_dir, llm_params)
            elapsed_time = time.time() - start_time
            logger.info(f"Elapsed time for processing {input_file}: {elapsed_time:.2f} seconds")

        logger.info("Introduction letter processing completed successfully.")
    except Exception as e:
        logger.error(f"Error in process_introduction_letters: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting script execution")

    # Create a configuration dictionary
    config = {
        'base_dir': base_dir,
        'output_dirs': {
            'Written_Introduction_Letter': OUTPUT_DIR
        }
    }

    # Process the introduction letter
    process_introduction_letters(INPUT_FILE, OUTPUT_DIR, config)

    logger.info("Script execution completed")