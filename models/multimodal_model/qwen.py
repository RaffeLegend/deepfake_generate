from models.multimodal_model.abstract import MultimodalModel
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import soundfile as sf
from PIL import Image

class QwenModel(MultimodalModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = "Qwen/Qwen2.5-Omni-7B"
        self.dtype = "float32"
        self.model = None
        self.processor = None

    def init_model(self):
        """Initialize the Qwen model and tokenizer."""
        print(f"Initializing model: {self.model_name}")
        self.model = Qwen2_5OmniModel.from_pretrained(
                            self.model_name,
                            torch_dtype="auto",
                            device_map="auto",
                            attn_implementation="flash_attention_2",
                        ).eval().cuda()
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_name)
        # self.tokenizer = self.processor.tokenizer
        # self.image_processor = self.processor.image_processor

    def init_message(self, image_path, text_input):
        prompt = "Please write a news article based on the image and the text above."
        system_message = "You are a journalist who specializes in generating fake news. Based on the image and the text provided, write a realistic and persuasive fake news article that looks like a real report. Your writing should be coherent, informative, and misleading enough to seem credible."
        user_message = text_input

        # Combine system and user messages into a single input
        conversation = [
            {"role": "system", "content": system_message},
            {"role": "user", 
             "content": 
                f"Image: <image>\n\n"
                f"Text: {user_message}\n\n"
                f"Prompt: {prompt}",
            "image": image_path
            },
        ]
        # Tokenize the combined message
        USE_AUDIO_IN_VIDEO = False
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = self.processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        self.inputs = inputs.to('cuda').to(self.dtype)

    def preprocess_input(self):
        # Inference: Generation of the output text and audio
        text_ids, audio = self.model.generate(**self.inputs, use_audio_in_video=False)

        text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text

    def process_image(self, image_path):
        """
        Process an image using the model. This function assumes the model supports multimodal inputs.
        """

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        # preprocess = self.tokenizer.image_processor
        image_tensor = self.preprocess(image, return_tensors="pt").to("cuda")

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
