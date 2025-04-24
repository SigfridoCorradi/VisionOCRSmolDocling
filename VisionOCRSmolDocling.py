import torch
import os
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

class VisionOCRSmolDocling:
    DOCLING_MODEL_ID: str = "ds4sd/SmolDocling-256M-preview"
    DOCLING_LOCAL_DIR: str = "./smoldocling_local_model"

    def __init__(self,
                 model_id: str = DOCLING_MODEL_ID,
                 local_dir: str = DOCLING_LOCAL_DIR,
                 device: str | None = None,
                 torch_dtype: torch.dtype = torch.bfloat16,
                 attention_implementation: str = "eager"):

        self.model_id = model_id
        self.local_dir = local_dir
        self.torch_dtype = torch_dtype
        self.attention_implementation = attention_implementation

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._download_model_if_needed()

        #Loading processor from self.local_dir
        self.processor = AutoProcessor.from_pretrained(self.local_dir)

        #Loading pre-trained model from self.local_dir
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.local_dir,
            torch_dtype=self.torch_dtype,
            _attn_implementation=self.attention_implementation,
        ).to(self.device)

    def _download_model_if_needed(self):
        #Checks if the model exists locally and downloads it if not.
        if not os.path.exists(self.local_dir):
            try:
                snapshot_download(
                    repo_id=self.model_id,
                    local_dir=self.local_dir,
                    local_dir_use_symlinks=False,
                )
            except Exception as e:
                print(f"Error downloading model: {e}")
                raise

    def extract_text(self,
                    image_path: str,
                    prompt_text: str = "Convert this page to docling.",
                    max_new_tokens: int = 8192) -> str:

        try:
            image = load_image(image_path)
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return "Error: Image " + image_path + " not found"
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return f"Error loading image " + image_path + ": {e}"

        # Defining Input Messages with prompt and image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            },
        ]

        try:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                text=prompt,
                images=[image],
                return_tensors="pt",
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        except Exception as e:
            print(f"Error processing inputs: {e}")
            return f"Error processing inputs: {e}"

        # Generating output
        raw_doctags = "Error during model generation" # Default error value
        try:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens
                )

            if 'input_ids' not in inputs:
                 print("Error during model generation") #'input_ids' not found in processor output.
                 return "Error during model generation"
            prompt_length = inputs['input_ids'].shape[1]

            trimmed_generated_ids = generated_ids[:, prompt_length:]
            raw_doctags = self.processor.batch_decode(
                trimmed_generated_ids,
                skip_special_tokens=False, #keep 'skip_special_tokens=False' as the raw output might contain tokens meaningful in the context of how Docling expects to parse them later, even if we don't parse here.
                truncation=True
            )[0].lstrip()

        except Exception as e:
            print(f"Error during model generation: {e}")
            raw_doctags = f"Error during model generation: {str(e)}"

        return raw_doctags

if __name__ == "__main__":
    IMAGE_TO_TEST = "test.jpg"

    if not os.path.exists(IMAGE_TO_TEST):
        print(f"Error: Test image '{IMAGE_TO_TEST}' not found.")
    else:
        extractor = VisionOCRSmolDocling()
        print(f"\n--- Processing Image: {IMAGE_TO_TEST} ---")
        extracted_text = extractor.extract_text(image_path=IMAGE_TO_TEST)

        print("\n--- Extraction Results ---")
        print(extracted_text)
