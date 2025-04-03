import os
import json

import torchvision.transforms as T

import base64
from io import BytesIO

from utils.tools import is_folder
from prompts.prompt import PROMPT_GENERATE_DESCRIPTION
from globals.define import IMAGENET_MEAN, IMAGENET_STD

from abc import ABC, abstractmethod

class MultimodalModel(ABC):
    """
    Abstract base class for multimodal tasks, supporting text, image, audio, and video inputs.
    """
    @abstractmethod
    def __init__(self, config):
        """
        Initialize the multimodal model.
        :param config: Configuration dictionary
        """
        self.prompt_set = None
        self.height = None
        self.width  = None
        self.model_name  = None
        self.model_path = None
        self.save_path = None
        self.model = None
        self.data_name = None
        self.save_size = None
        self.image_format = None

        self.task_type = None  # Define the type of task (e.g., 'generation', 'detection', etc.)
        self.output_type = None  # Define the type of output (e.g., 'image', 'video', etc.)

    # Set the output path for saving results
    def get_save_path(self, output_path: str, data_name: str) -> None:
        self.data_name = data_name
        self.save_path = os.path.join(output_path, f"{self.model_name}_output")
        is_folder(self.save_path)  # Ensure the folder exists
    
    # Save JSON file
    def save_json(self, data: dict, index: int) -> None:
        file_name = f"{self.data_name}_{str(index).zfill(4)}.json"
        json_path = os.path.join(self.save_path, file_name)
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    def get_images_path(self, set_path: str, save_size: int, image_format: str) -> None:
        """
        Retrieve image paths and save metadata in JSON files.
        :param set_path: Path to the dataset folder
        :param save_size: Number of entries per JSON file
        :param image_format: Image file format (e.g., '.jpg', '.png')
        """
        self.save_size = save_size
        self.image_format = image_format
        file_paths = []
        index_file = 0

        for root, _, files in os.walk(set_path):
            image_files = [file for file in files if file.endswith(self.image_format)]
            for file in image_files:
                file_paths.append({
                    "index": str(len(file_paths)).zfill(9),
                    "file_name": file,
                    "file_path": root,
                    "prompt": "",
                    "text": ""
                })
                if len(file_paths) >= self.save_size:
                    self.save_json(file_paths, index_file)
                    index_file += 1
                    file_paths = []

        if file_paths:
            self.save_json(file_paths, index_file)

    def add_text_to_images(self, text_file: str) -> None:
        """
        Add text descriptions to image metadata in JSON files.
        :param text_file: Path to the text file containing descriptions
        """
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"Text file {text_file} not found.")

        # Read text descriptions from the file
        with open(text_file, "r", encoding="utf-8") as file:
            descriptions = file.readlines()

        # Iterate over JSON files in the save path
        for json_file in os.listdir(self.save_path):
            if json_file.endswith(".json"):
                json_path = os.path.join(self.save_path, json_file)
                
                # Load the JSON data
                with open(json_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                # Add text descriptions to each image entry
                for i, entry in enumerate(data):
                    if i < len(descriptions):
                        entry["text"] = descriptions[i].strip()
                    else:
                        entry["text"] = ""

                # Save the updated JSON data
                with open(json_path, "w", encoding="utf-8") as file:
                    json.dump(data, file, indent=4, ensure_ascii=False)

    # Load data from JSON files in the save path
    def load_data(self):
        """
        Load metadata from all JSON files in the save path.
        :return: List of file paths to the JSON files
        """
        return [
            os.path.join(root, file)
            for root, _, files in os.walk(self.save_path)
            for file in files if file.endswith(".json")
        ]

    def load_image_text_pairs(self, image_path: str, text_path: str):
        """
        Load image and text pairs from a JSON file and save them to a new JSON file.
        :param image_path: Path to the JSON file
        :param text_path: Path to the JSON file containing text descriptions
        """
        with open(text_path, "r", encoding="utf-8") as infile:
            data = json.load(infile)
        
        image_text_pairs = [
            {"image_path": os.path.join(entry["file_path"], entry["file_name"]), "text": entry["text"]}
            for entry in data
        ]

        output_file = os.path.join(self.save_path, "image_text_pairs.json")
        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(image_text_pairs, outfile, indent=4, ensure_ascii=False)

    @abstractmethod
    def process_text(self, text):
        """
        Process text input.
        :param text: str
        :return: Processed text output
        """
        pass

    @abstractmethod
    def process_image(self, image):
        """
        Process image input.
        :param image: Image data (e.g., numpy array or PIL Image)
        :return: Processed image output
        """
        pass

    @abstractmethod
    def process_audio(self, audio):
        """
        Process audio input.
        :param audio: Audio data (e.g., waveform or spectrogram)
        :return: Processed audio output
        """
        pass

    @abstractmethod
    def process_video(self, video):
        """
        Process video input.
        :param video: Video data (e.g., frames or video file)
        :return: Processed video output
        """
        pass

    @abstractmethod
    def forward(self, inputs):
        """
        Forward pass for multimodal inputs.
        :param inputs: Dictionary containing multimodal inputs (e.g., {'text': ..., 'image': ..., 'audio': ..., 'video': ...})
        :return: Model output
        """
        pass