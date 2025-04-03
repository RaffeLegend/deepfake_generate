from .abstract import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

class QwenModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"

    def init_model(self):
        """Initialize the Qwen model and tokenizer."""
        print(f"Initializing model: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def init_message(self):
        prompt = "Provide a brief introduction to large language models."
        system_message = "You are Qwen, an AI assistant created by Alibaba Cloud. Your purpose is to assist users effectively."
        user_message = prompt

        # Combine system and user messages into a single input
        combined_message = f"{system_message}\n\nUser: {user_message}\n\nAssistant:"
        
        # Tokenize the combined message
        self.model_inputs = self.tokenizer(
            combined_message, return_tensors="pt", padding=True, truncation=True
        ).to("cuda")

    def preprocess_input(self):
        # Generate response from the model
        generated_ids = self.model.generate(
            **self.model_inputs,
            max_new_tokens=512
        )
        
        # Extract the generated portion of the response
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(self.model_inputs.input_ids, generated_ids)
        ]
        
        # Decode the generated tokens into text
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Return the processed response
        return response

    def process_image(self, image_path):
        """
        Process an image using the model. This function assumes the model supports multimodal inputs.
        """

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        preprocess = self.tokenizer.image_processor
        image_tensor = preprocess(image, return_tensors="pt").to("cuda")

        # Generate response using the model
        outputs = self.model.generate(
            pixel_values=image_tensor["pixel_values"],
            max_new_tokens=512
        )

        # Decode the generated tokens into text
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # Return the processed response
        return response
    
    def process_image_and_text(self, image_path, text_input):
        """
        Process a pair of image and text using the model. This function assumes the model supports multimodal inputs.
        """
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        preprocess = self.tokenizer.image_processor
        image_tensor = preprocess(image, return_tensors="pt").to("cuda")

        # Tokenize the text input
        text_inputs = self.tokenizer(
            text_input, return_tensors="pt", padding=True, truncation=True
        ).to("cuda")

        # Generate response using the model
        outputs = self.model.generate(
            pixel_values=image_tensor["pixel_values"],
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            max_new_tokens=512
        )

        # Decode the generated tokens into text
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # Return the processed response
        return response