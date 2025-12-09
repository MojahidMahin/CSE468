"""
VLM Model Wrapper Module

Provides unified interface for multiple Vision-Language Models.
Each model class handles its own initialization, inference, and cleanup.
"""

import torch
import gc
from PIL import Image
from typing import Optional
from abc import ABC, abstractmethod
from config import EvaluationConfig


class BaseVLMModel(ABC):
    """Base class for all VLM models"""

    def __init__(self, model_id: str, device: str = "cuda", use_fp16: bool = True):
        self.model_id = model_id
        self.device = device
        self.use_fp16 = use_fp16
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self):
        """Load the model and processor"""
        pass

    @abstractmethod
    def generate_caption(self, image: Image.Image, prompt: str) -> str:
        """Generate caption for an image"""
        pass

    def unload_model(self):
        """Unload model and free memory"""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"✓ {self.__class__.__name__} unloaded from memory")


class Qwen2VLModel(BaseVLMModel):
    """Qwen2-VL-2B-Instruct model wrapper"""

    def load_model(self):
        """Load Qwen2-VL model"""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        print(f"Loading {self.model_id}...")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
            trust_remote_code=True
        )

        print(f"✓ {self.model_id} loaded successfully")

    def generate_caption(self, image: Image.Image, prompt: str) -> str:
        """Generate caption using Qwen2-VL"""
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        caption = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return caption.strip()


class Qwen3VLModel(BaseVLMModel):
    """Qwen3-VL model wrapper (newer architecture)"""

    def load_model(self):
        """Load Qwen3-VL model"""
        from transformers import AutoModel, AutoProcessor
        from qwen_vl_utils import process_vision_info

        print(f"Loading {self.model_id}...")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
            trust_remote_code=True
        )

        print(f"✓ {self.model_id} loaded successfully")

    def generate_caption(self, image: Image.Image, prompt: str) -> str:
        """Generate caption using Qwen3-VL"""
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        caption = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return caption.strip()


class Phi3VisionModel(BaseVLMModel):
    """Phi-3-Vision model wrapper"""

    def load_model(self):
        """Load Phi-3-Vision model"""
        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"Loading {self.model_id}...")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
            trust_remote_code=True,
            _attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )

        print(f"✓ {self.model_id} loaded successfully")

    def generate_caption(self, image: Image.Image, prompt: str) -> str:
        """Generate caption using Phi-3-Vision"""
        messages = [
            {
                "role": "user",
                "content": f"<|image_1|>\n{prompt}"
            }
        ]

        prompt_text = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            prompt_text,
            [image],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )

        generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]

        caption = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return caption.strip()


class InternVL2Model(BaseVLMModel):
    """InternVL2-2B model wrapper"""

    def load_model(self):
        """Load InternVL2 model"""
        from transformers import AutoModel, AutoTokenizer

        print(f"Loading {self.model_id}...")

        self.model = AutoModel.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
            trust_remote_code=True
        ).eval()

        self.processor = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        print(f"✓ {self.model_id} loaded successfully")

    def generate_caption(self, image: Image.Image, prompt: str) -> str:
        """Generate caption using InternVL2"""
        # Load image using model's load_image function
        from transformers import load_image

        # Save PIL image temporarily and load it back (workaround for InternVL2)
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Use InternVL2's expected image loading
            pixel_values = load_image(tmp_path, max_num=12).to(
                torch.float16 if self.use_fp16 else torch.float32
            ).to(self.device)

            generation_config = dict(
                max_new_tokens=256,
                do_sample=False
            )

            with torch.no_grad():
                response = self.model.chat(
                    self.processor,
                    pixel_values,
                    prompt,
                    generation_config
                )
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return response.strip()


class SmolVLM2Model(BaseVLMModel):
    """SmolVLM2-Instruct model wrapper"""

    def load_model(self):
        """Load SmolVLM2 model"""
        from transformers import AutoProcessor, AutoModelForVision2Seq

        print(f"Loading {self.model_id}...")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
            trust_remote_code=True,
            _attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )

        print(f"✓ {self.model_id} loaded successfully")

    def generate_caption(self, image: Image.Image, prompt: str) -> str:
        """Generate caption using SmolVLM2"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        prompt_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=prompt_text,
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        # Extract only the assistant's response
        caption = generated_texts[0].split("Assistant:")[-1].strip()

        return caption


class VLMModelFactory:
    """Factory class to create VLM models"""

    MODEL_MAP = {
        "Qwen2-VL-2B": Qwen2VLModel,
        "Qwen3-VL-4B": Qwen3VLModel,
        "Qwen2.5-VL-7B": Qwen2VLModel,
        "Phi3-Vision": Phi3VisionModel,
        "InternVL2-2B": InternVL2Model,
        "SmolVLM2": SmolVLM2Model
    }

    @classmethod
    def create_model(cls, model_name: str, model_id: str, device: str = "cuda", use_fp16: bool = True):
        """Create and return a VLM model instance"""
        if model_name not in cls.MODEL_MAP:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.MODEL_MAP.keys())}")

        model_class = cls.MODEL_MAP[model_name]
        return model_class(model_id=model_id, device=device, use_fp16=use_fp16)


if __name__ == "__main__":
    # Test VLM model wrapper
    print("Testing VLM Model Wrapper Module...\n")

    try:
        from data_loader import RadiologydataLoader

        # Load a test image
        loader = RadiologydataLoader()
        sample = loader.get_sample(0)
        test_image = sample['image']
        test_prompt = EvaluationConfig.ZERO_SHOT_PROMPT

        print(f"Test image: {sample['id']}")
        print(f"Ground truth: {sample['caption'][:100]}...\n")

        # Test Qwen2-VL model (smallest, fastest to test)
        print("Testing Qwen2-VL-2B model...")
        model = VLMModelFactory.create_model(
            model_name="Qwen2-VL-2B",
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            device=EvaluationConfig.DEVICE,
            use_fp16=EvaluationConfig.USE_FP16
        )

        model.load_model()
        caption = model.generate_caption(test_image, test_prompt)
        print(f"Generated caption: {caption}\n")

        model.unload_model()

        print("✓ VLM model wrapper test passed!")

    except Exception as e:
        print(f"\n✗ VLM model wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
