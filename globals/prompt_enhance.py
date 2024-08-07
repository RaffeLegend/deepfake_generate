import torch
from transformers import GenerationConfig, GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessor, LogitsProcessorList

styles = {
    "cinematic": "cinematic film still of {prompt}, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain",
    "anime": "anime artwork of {prompt}, anime style, key visual, vibrant, studio anime, highly detailed",
    "photographic": "cinematic photo of {prompt}, 35mm photograph, film, professional, 4k, highly detailed",
    "comic": "comic of {prompt}, graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
    "lineart": "line art drawing {prompt}, professional, sleek, modern, minimalist, graphic, line art, vector graphics",
    "pixelart": " pixel-art {prompt}, low-res, blocky, pixel art style, 8-bit graphics",
}

words = [
    "aesthetic", "astonishing", "beautiful", "breathtaking", "composition", "contrasted", "epic", "moody", "enhanced",
    "exceptional", "fascinating", "flawless", "glamorous", "glorious", "illumination", "impressive", "improved",
    "inspirational", "magnificent", "majestic", "hyperrealistic", "smooth", "sharp", "focus", "stunning", "detailed",
    "intricate", "dramatic", "high", "quality", "perfect", "light", "ultra", "highly", "radiant", "satisfying",
    "soothing", "sophisticated", "stylish", "sublime", "terrific", "touching", "timeless", "wonderful", "unbelievable",
    "elegant", "awesome", "amazing", "dynamic", "trendy",
]

word_pairs = ["highly detailed", "high quality", "enhanced quality", "perfect composition", "dynamic light"]

def find_and_order_pairs(s, pairs):
    words = s.split()
    found_pairs = []
    for pair in pairs:
        pair_words = pair.split()
        if pair_words[0] in words and pair_words[1] in words:
            found_pairs.append(pair)
            words.remove(pair_words[0])
            words.remove(pair_words[1])

    for word in words[:]:
        for pair in pairs:
            if word in pair.split():
                words.remove(word)
                break
    ordered_pairs = ", ".join(found_pairs)
    remaining_s = ", ".join(words)
    return ordered_pairs, remaining_s

class CustomLogitsProcessor(LogitsProcessor):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def __call__(self, input_ids, scores):
        if len(input_ids.shape) == 2:
            last_token_id = input_ids[0, -1]
            self.bias[last_token_id] = -1e10
        return scores + self.bias


class PromptEnhancer:
    def __init__(self):
        super().__init__()
        self.model_path     = "Gustavosta/MagicPrompt-Stable-Diffusion"
        self.torch_dtype    = torch.float16
        self.processor_list = None
        self.generated_ids  = None
        self.max_new_tokens = 0

    def __call__(self, prompt, style):
        self.init_model()
        self.init_processor_list()
        self.prompt_preprocess(prompt, style)
        self.model_generate()
        enhanced_prompt = self.prompt_enhanced()

        return enhanced_prompt

    def init_model(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
        self.model = GPT2LMHeadModel.from_pretrained(
                            self.model_path, 
                            torch_dtype=self.torch_dtype
                            ).to("cuda")
        self.model.eval()

    def init_processor_list(self):
        word_ids = [self.tokenizer.encode(word, add_prefix_space=True)[0] for word in words]
        bias = torch.full((self.tokenizer.vocab_size,), -float("Inf")).to("cuda")
        bias[word_ids] = 0
        processor = CustomLogitsProcessor(bias)
        self.processor_list = LogitsProcessorList([processor])

    def set_generation_config(self):
        self.generation_config = GenerationConfig(
                            penalty_alpha=0.7,
                            top_k=50,
                            eos_token_id=self.model.config.eos_token_id,
                            pad_token_id=self.model.config.eos_token_id,
                            pad_token=self.model.config.pad_token_id,
                            do_sample=True,
                            )
        
    def model_generate(self):
        with torch.no_grad():
            self.generated_ids = self.model.generate(
                            input_ids=self.inputs["input_ids"],
                            attention_mask=self.inputs["attention_mask"],
                            max_new_tokens=self.max_new_tokens,
                            generation_config=self.generation_config,
                            logits_processor=self.processor_list,
                            )
            
    def prompt_preprocess(self, prompt, style):
        self.prompt = styles[style].format(prompt=prompt)
        self.inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        token_count = self.inputs["input_ids"].shape[1]
        self.max_new_tokens = 50 - token_count

    def prompt_enhanced(self):
        output_tokens = [self.tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in self.generated_ids]
        input_part, generated_part = output_tokens[0][: len(self.prompt)], output_tokens[0][len(self.prompt) :]
        pairs, words = find_and_order_pairs(generated_part, word_pairs)
        formatted_generated_part = pairs + ", " + words
        enhanced_prompt = input_part + ", " + formatted_generated_part

        return enhanced_prompt
