"""Local BF16 adapters. Model-native preprocessing is recorded, never hidden."""
from pathlib import Path

from PIL import Image


class Adapter:
    def __init__(self, family, path):
        import torch
        from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
        self.torch = torch
        self.family = family
        self.path = Path(path)
        kwargs = dict(trust_remote_code=True, local_files_only=True,
                      torch_dtype=torch.bfloat16, device_map='cuda:0')
        if family in ('minicpm', 'internvl'):
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
            if family == 'internvl':
                kwargs['use_flash_attn'] = False
            else:
                kwargs['attn_implementation'] = 'sdpa'
            self.model = AutoModel.from_pretrained(path, **kwargs).eval()
            if family == 'internvl':
                self.model.language_model.set_attn_implementation('sdpa')
        else:
            self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
            kwargs['attn_implementation'] = 'sdpa'
            self.model = AutoModelForImageTextToText.from_pretrained(path, **kwargs).eval()

    def generate(self, task, max_tokens):
        torch = self.torch
        images = []
        for name in task['images']:
            with Image.open(name) as im:
                images.append(im.convert('RGB'))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            if self.family == 'minicpm':
                slices = self.model.config.slice_config.max_slice_nums
                raw = self.model.chat(msgs=[{'role': 'user', 'content': images + [task['prompt_text']]}],
                    tokenizer=self.tokenizer, system_prompt=task['system_prompt'],
                    max_new_tokens=max_tokens, sampling=False, enable_thinking=False,
                    num_beams=1, repetition_penalty=1.0,
                    max_inp_length=self.model.config.max_position_embeddings,
                    max_slice_nums=slices, use_image_id=True)
                return str(raw), {'max_slice_nums': slices, 'thinking': False,
                                  'num_beams': 1, 'repetition_penalty': 1.0,
                                  'input_limit': self.model.config.max_position_embeddings}
            if self.family == 'internvl':
                from torchvision import transforms
                from torchvision.transforms.functional import InterpolationMode
                transform = transforms.Compose([
                    transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                    transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
                # Match the existing InternVL baseline's one-patch-per-image adapter.
                pixels = torch.stack([transform(im) for im in images]).to('cuda', dtype=torch.bfloat16)
                prefix = ''.join(f'Image-{i + 1}: <image>\n' for i in range(len(images)))
                question = task['system_prompt'] + '\n' + prefix + task['prompt_text']
                raw = self.model.chat(self.tokenizer, pixels, question,
                    {'max_new_tokens': max_tokens, 'do_sample': False},
                    num_patches_list=[1] * len(images), history=None, return_history=False)
                return str(raw), {'image_size': 448, 'patches_per_image': 1,
                                  'attention_backend': 'sdpa', 'prompt_suffix': None}
            content = [{'type': 'image', 'image': image} for image in images]
            # Molmo's official template only accepts alternating user/assistant roles.
            text = task['prompt_text']
            if self.family == 'molmo':
                text = task['system_prompt'] + '\n\n' + text
            content.append({'type': 'text', 'text': text})
            messages = [{'role': 'user', 'content': content}]
            if self.family != 'molmo':
                messages.insert(0, {'role': 'system', 'content': task['system_prompt']})
            if self.family == 'onevision':
                # This custom processor delegates templates to the text tokenizer;
                # images must be supplied explicitly in a second call.
                rendered = self.processor.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                inputs = self.processor(text=[rendered], images=images, return_tensors='pt', padding=True)
            else:
                inputs = self.processor.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors='pt')
            if 'pixel_values' not in inputs:
                raise RuntimeError('Processor omitted visual tensors; refusing text-only inference')
            inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in inputs.items()}
            input_length = inputs['input_ids'].shape[-1]
            result = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            raw = self.processor.tokenizer.decode(result[0, input_length:], skip_special_tokens=True)
            return raw, {'input_tokens': input_length, 'output_tokens': result.shape[-1] - input_length,
                         'thinking': False, 'processor': type(self.processor).__name__}
