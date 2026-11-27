"""Native Qwen3.5 multimodal generation; preserve the existing task protocol."""
from pathlib import Path
from PIL import Image


class Qwen35Adapter:
    def __init__(self, family, path):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16, device_map='cuda:0',
            attn_implementation='sdpa', local_files_only=True).eval()

    def generate(self, task, max_tokens):
        images = []
        for name in task['images']:
            with Image.open(name) as image:
                images.append(image.convert('RGB'))
        content = [{'type': 'image', 'image': image} for image in images]
        content.append({'type': 'text', 'text': task['prompt_text']})
        messages = [{'role': 'system', 'content': task['system_prompt']},
                    {'role': 'user', 'content': content}]
        # Official non-thinking mode matches the original direct-answer baseline.
        inputs = self.processor.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors='pt')
        if 'pixel_values' not in inputs:
            raise RuntimeError('Qwen3.5 processor omitted visual tensors')
        inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in inputs.items()}
        length = inputs['input_ids'].shape[-1]
        with self.torch.inference_mode(), self.torch.autocast('cuda', dtype=self.torch.bfloat16):
            output = self.model.generate(**inputs, max_new_tokens=max_tokens,
                do_sample=False, num_beams=1, use_cache=True)
        return self.processor.tokenizer.decode(output[0, length:], skip_special_tokens=True), {
            'input_tokens': length, 'output_tokens': output.shape[-1] - length,
            'thinking': False, 'processor': type(self.processor).__name__,
            'image_grid_thw': inputs['image_grid_thw'].tolist(),
            'attention_backend': 'sdpa', 'native_image_processing': True}
