import json
import sys
import os
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.multimodal_model.qwen import QwenModel

def read_and_save_json(input_file, output_file):
    """
    Reads a JSON file, extracts 'text' and 'image_path' fields, 
    and writes the filtered content to another file.

    :param input_file: Path to the input JSON file.
    :param output_file: Path to the output JSON file.
    """
    try:
        # Read JSON data from the input file
        with open(input_file, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        # Extract only 'text' and 'image_path' fields
        filtered_data = [
            {"text": item.get("text"), "image_path": item.get("image_path")}
            for item in data if "text" in item and "image_path" in item
        ]
        
        # Write filtered JSON data to the output file
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(filtered_data, outfile, indent=4, ensure_ascii=False)
        
        print(f"Filtered JSON data successfully saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_with_qwen_model(json_file):
    """
    Processes each entry in the JSON file using the Qwen model.

    :param json_file: Path to the JSON file containing 'text' and 'image_path'.
    :param model_name: Name of the Qwen model to use.
    """
    # Initialize the Qwen model
    qwen_model = QwenModel(config="")
    qwen_model.init_model()
    try:
        # Read the JSON data once
        with open(json_file, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        # Process each entry with the Qwen model
        for item in data:
            text = item.get("text", "")
            image_path = "/mnt/data1/users/yiwei/data/mmfakebench/MMFakeBench_test"+item.get("image_path", "")
            # Skip processing if either text or image_path is missing
            if not text or not image_path:
                continue
            # Process the text and image using the Qwen model
            qwen_model.init_message(image_path, text)
            response = qwen_model.preprocess_input()
            # Add the response to the current item
            item["response"] = response

        # Save the updated JSON data back to the file once
        with open(json_file, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)

        print(f"Processed JSON data successfully updated in {json_file}")
    except Exception as e:
        print(f"An error occurred while processing with Qwen model: {e}")
        traceback.print_exc(file=sys.stdout)

# Example usage
if __name__ == "__main__":
    input_path = "/mnt/data1/users/yiwei/data/mmfakebench/MMFakeBench_test.json"  # Replace with your input JSON file path
    output_path = "output.json"  # Replace with your desired output JSON file path
    read_and_save_json(input_path, output_path)
    process_with_qwen_model(output_path)
