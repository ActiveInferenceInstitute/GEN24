import os
import logging
from utils import load_file, save_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

#  Write INTRODUCTION_LETTER between ENTITY_1 and ENTITY_2, who work at ORG_1, from ENTITY_3 introducer from ORG_2.

ENTITY_1_FILE = os.path.join(SCRIPT_DIR, 'Inputs_and_Outputs', 'Entity', 'Hipolito', 'Hipolito.py')
ENTITY_2_FILE = os.path.join(SCRIPT_DIR, 'Inputs_and_Outputs', 'Entity', 'Friston', 'Friston.py')
ENTITY_3_FILE = os.path.join(SCRIPT_DIR, 'Inputs_and_Outputs', 'Entity', 'Shyaka', 'Shyaka.py')
ORG_1_FILE = os.path.join(SCRIPT_DIR, 'Inputs_and_Outputs', 'Organization', 'Active_Inference_Institute', 'Active_Inference_Institute.py')
ORG_2_FILE = os.path.join(SCRIPT_DIR, 'Inputs_and_Outputs', 'Organization', 'AIME', 'AIME.py')

def generate_introduction_letter(entity1_file, entity2_file, entity3_file, org1_file, org2_file, prompt, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info("Created output directory: %s", output_dir)
    
    try:
        entity1_data = load_file(entity1_file)
        entity2_data = load_file(entity2_file)
        entity3_data = load_file(entity3_file)
        org1_data = load_file(org1_file)
        org2_data = load_file(org2_file)
        
        entity1_name = os.path.splitext(os.path.basename(entity1_file))[0]
        entity2_name = os.path.splitext(os.path.basename(entity2_file))[0]
        entity3_name = os.path.splitext(os.path.basename(entity3_file))[0]
        org1_name = os.path.splitext(os.path.basename(org1_file))[0]
        org2_name = os.path.splitext(os.path.basename(org2_file))[0]
        letter_content = f"""
Your task is to write a comprehensive, professional, concise Introduction Letter between {entity1_name} and {entity2_name}, who work at {org1_name}, from {entity3_name}, the introducer from {org2_name}.

1. Identify key quotes, perspectives, stances, beliefs, and insights from {entity1_name} and {entity2_name} at {org1_name}, ensuring these quotes represent their distinct yet complementary approaches within the shared organizational context.

2. For each quotes from each entity:
   a. Illuminate how {entity1_name}'s perspective enhances or provides new depth to {entity2_name}'s work within their institutional framework (and vice versa).
   b. Synthesize their combined perspectives to reveal a novel insight that emerges from their institutional collaboration.

3. After each synthesis of {entity1_name} and {entity2_name}'s ideas:
   a. Introduce a poignant quote or transformative concept from {entity3_name} or {org2_name}.
   b. Articulate how {entity3_name} or {org2_name}'s approach doesn't merely triangulate, but catalyzes a profound reimagining of {entity1_name} and {entity2_name}'s combined insights and their work at {org1_name}.
   c. Elucidate how this catalyzation unveils hidden potential and opens new dimensions in addressing the issue at hand.

4. For each triad of ideas:
   a. Paint a vivid picture of the comprehensive, multifaceted understanding that blossoms from the integration of all three perspectives.
   b. Explore how the institutional work of {entity1_name} and {entity2_name}, when illuminated by {entity3_name}'s approach, transcends its original scope and impact.
   c. Highlight how {entity3_name}'s contribution honors and elevates the institutional work, while pointing towards future possibilities not previously envisioned.

5. Weave throughout:
   a. An acknowledgment that the fuller potential of {entity1_name} and {entity2_name}'s work may be realized through the transformative lens provided by {entity3_name} or {org2_name}.
   b. Recognition that this triangulation doesn't diminish the importance of {entity1_name} and {entity2_name}'s work, but rather amplifies its impact and relevance.
   c. An exploration of how this three-way integration creates a harmonious symphony of ideas, each part essential to the whole.

6. Conclude by:
   a. Articulating an inspiring actionable vision that emerges from this intricate weave of institutional expertise and innovative social approach, especially in context of {org2_name}.
   b. Proposing many specific ideas, in bulletpoint list, where this triadic collaboration could lead to groundbreaking solutions or paradigm shifts.
   c. Emphasizing the transformative journey that lies ahead when these diverse yet harmonious perspectives unite in purpose.

Here is {entity1_name}:
{entity1_data}

Here is {entity2_name}:
{entity2_data}

Here is {entity3_name}:
{entity3_data}

Here is {org1_name}:
{org1_data}

Here is {org2_name}:
{org2_data}

Refer to the above information as needed to complete the letter, write it professionally from {entity3_name} who works at {org2_name} now: 
"""
        output_file_name = f"introduction_letter_{entity1_name}_and_{entity2_name}.md"
        output_file_path = os.path.join(output_dir, output_file_name)
        save_file(letter_content, output_file_path)
        logger.info("Saved introduction letter to %s", output_file_path)
    except Exception as e:
        logger.error("Failed to generate introduction letter: %s", str(e))

def main():
    prompt_path = os.path.join(SCRIPT_DIR, 'Inputs_and_Outputs', 'Prompts', 'Introduction_Letter_Prompt.md')
    output_dir = os.path.join(SCRIPT_DIR, 'Inputs_and_Outputs', 'Pro-Introduction_Letter')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info("Created output directory: %s", output_dir)

    prompt = load_file(prompt_path)
    logger.info("Output directory: %s", output_dir)

    generate_introduction_letter(ENTITY_1_FILE, ENTITY_2_FILE, ENTITY_3_FILE, ORG_1_FILE, ORG_2_FILE, prompt, output_dir)

    logger.info("Process completed. Check logs for integration results.")

if __name__ == "__main__":
    main()