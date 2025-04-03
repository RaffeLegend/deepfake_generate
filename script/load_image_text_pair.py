import json

def read_and_save_json(input_file, output_file):
    """
    Reads a JSON file and writes its content to another file.

    :param input_file: Path to the input JSON file.
    :param output_file: Path to the output JSON file.
    """
    try:
        # Read JSON data from the input file
        with open(input_file, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        # Write JSON data to the output file
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)
        
        print(f"JSON data successfully saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    input_path = "input.json"  # Replace with your input JSON file path
    output_path = "output.json"  # Replace with your desired output JSON file path
    read_and_save_json(input_path, output_path)