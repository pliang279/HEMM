import abc
import torch
from typing import Optional, Union, List
from PIL import Image
from huggingface_hub import hf_hub_download
import torch
import re
from tqdm import tqdm

from hemm.models.model import HEMMModel
from hemm.utils.common_utils import shell_command

from hemm.models.open_flamingo.open_flamingo import create_model_and_transforms
from hemm.models.open_flamingo.open_flamingo.eval.utils import unwrap_model, get_autocast, get_cast_dtype

def ref_text(text):
    sents = text.split("\n")
    sents = [sent.strip() for sent in sents]
    return " ".join(sents).strip()

class OpenFlamingoModel(HEMMModel):
    def __init__(self, device="cuda", download_dir="./", **kwargs) -> None:
        self.device = torch.device(device)
        self.cache_dir = download_dir

    def load_weights(self):
        """
        Loads the model and it's processor with the initialized weight directory
        :return:
        """
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1,
            cache_dir=self.cache_dir,
        )

        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        if "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

        self.model.load_state_dict(ckpt, strict=False)
        self.tokenizer.padding_side = "left"
        self.model = self.model.to(self.device)
    
    def get_image_tensor(self, image):
        """
        Get image tensor using model's vision processor.
        Tensor should have batch dimension.
        """
        vision_x = self.image_processor(image).unsqueeze(0)
        return vision_x

    def generate(self,
                 text: Optional[str],
                 image
                 ):
        """
        Generates output for the given prompt.
        :param text: String text prompt
        :param image: Image prompt (pillow)
        :param return_logits: return logit tensor if true, else return text/image.
        :return: return str, image or logit tensor.
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        # image = Image.open(image)
        vision_x = [self.image_processor(image).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(self.device)

        lang_x = self.tokenizer(
            ["<image>" + str(text) + "<|endofchunk|>"],
            return_tensors="pt",
        ).to(self.device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            generated_text = self.model.generate(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                max_new_tokens=50,
            )

        from_len = len(lang_x["input_ids"][0])
        decoded_text =  self.tokenizer.decode(generated_text[0][from_len:], skip_special_tokens=True)
        decoded_text = decoded_text.replace(ref_text(text), '')
        decoded_text = decoded_text.replace('<image>', '')
        return decoded_text

    def generate_batch_func(self, batch_images, batch_texts, batch_size, max_new_tokens=50, num_beams=3):
        """
        batch_images: [8,3,512,512]
        batch_texts: [str1, str2, str3, str4,str1, str6, str7, str8,]
        """
        texts = ['<image> '+ str(text_) for text_ in batch_texts]
        encodings = self.tokenizer(
            texts,
            padding='longest',
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(
            self.device, non_blocking=True
        )
        attention_mask = attention_mask.bool()
        batch_images = batch_images.unsqueeze(1).unsqueeze(1) 
        batch_images.to(self.device)

        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model.generate(
                    batch_images,
                    input_ids,
                    attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams
                )

        outputs = outputs[:, len(input_ids[0]) :]
        out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return out

    def generate_batch(self, images, texts, batch_size, max_new_tokens=50, num_beams=3):
        answers = []
        total_batches = len(texts) // batch_size

        for batch_idx in tqdm(range(total_batches), total=total_batches):
            start_idx = batch_idx * batch_size 
            end_idx = (batch_idx + 1) * batch_size 
            
            batch_texts = texts[start_idx:end_idx]
            batch_images = images[start_idx:end_idx].to(self.device)
            batch_answers = self.generate_batch_func(batch_images, batch_texts, batch_size, max_new_tokens, num_beams)
            answers.extend(batch_answers)
            
        if len(texts) % batch_size != 0:
            start_idx = total_batches * batch_size
            end_idx = len(texts)

            # Get the remaining data as a smaller batch
            batch_texts = texts[start_idx:end_idx]
            batch_images = images[start_idx:end_idx].to(self.device)
            batch_answers = self.generate_batch_func(batch_images, batch_texts, batch_size, max_new_tokens, num_beams)
            answers.extend(batch_answers)

        return answers

    def answer_extractor(self, text, dataset_key):
        if dataset_key == 'hateful_memes' or dataset_key =='newyorkercartoon' or dataset_key =='irfl':
            text = text[:3]
            text = text.lower().strip()
            text = ''.join(filter(str.isalpha, text.lower()))
            return text
        elif dataset_key == 'memotion' or dataset_key == 'face_emotion'  or dataset_key == 'scienceqa' or dataset_key == 'vcr':
            match = re.search(r"\b\d\b", text)
            if match:
                first_number = int(match.group())
                return first_number
            else:
                return None
            