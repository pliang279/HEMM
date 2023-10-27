import os
import subprocess
import argparse
import re
import torch
from tqdm import tqdm
import time

from hemm.models.minigpt4.common.config import Config
from hemm.models.minigpt4.common.dist_utils import get_rank
from hemm.models.minigpt4.common.registry import registry
from hemm.models.minigpt4.conversation.conversation import Chat, CONV_VISION
from hemm.utils.common_utils import shell_command

from hemm.models.model import HEMMModel

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args(["--cfg-path", "/home/agoindan/HEMM/hemm/models/minigpt4/configs/eval_configs/minigpt4_eval.yaml"])
    return args

class MiniGPT4(HEMMModel):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_weights(self):
        # if not os.path.exists('MiniGPT-4'):
        #     subprocess.Popen('git clone https://github.com/Vision-CAIR/MiniGPT-4.git', shell=True)
        if not os.path.exists('prerained_minigpt4_7b.pth'):
            shell_command('wget https://huggingface.co/wangrongsheng/MiniGPT4-7B/resolve/main/prerained_minigpt4_7b.pth')

        # Read in the file
        with open('/home/agoindan/HEMM/hemm/models/minigpt4/configs/models/minigpt4.yaml', 'r') as file:
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace('/path/to/vicuna/weights/', 'wangrongsheng/MiniGPT-4-LLaMA-7B')

        # Write the file out again
        with open('/home/agoindan/HEMM/hemm/models/minigpt4/configs/models/minigpt4.yaml', 'w') as file:
            file.write(filedata)

        # Read in the file
        with open('/home/agoindan/HEMM/hemm/models/minigpt4/configs/eval_configs/minigpt4_eval.yaml', 'r') as file:
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace('/path/to/pretrained/ckpt/', 'prerained_minigpt4_7b.pth')

        # Write the file out again
        with open('/home/agoindan/HEMM/hemm/models/minigpt4/configs/eval_configs/minigpt4_eval.yaml', 'w') as file:
            file.write(filedata)
        
        args = parse_args()
        cfg = Config(args)

        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    
    def upload_img(self, gr_img, chat_state):
        chat_state = CONV_VISION.copy()
        img_list = []
        llm_message = self.chat.upload_img(gr_img, chat_state, img_list)
        return chat_state, img_list
    
    def ask_message(self, user_message, chat_state):
        if len(user_message) == 0:
            print("Input should not be empty!")
            # return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
        self.chat.ask(user_message, chat_state)
        # chatbot = chatbot + [[user_message, None]]
        return chat_state

    def answer_question(self, chat_state, img_list, num_beams, temperature):
        llm_message = self.chat.answer(conv=chat_state,
                                img_list=img_list,
                                num_beams=num_beams,
                                temperature=temperature,
                                max_new_tokens=300,
                                max_length=2000)[0]
        # print(llm_message)
        return llm_message, chat_state, img_list
    
    def generate(self, text, image):
        chat_state, img_list = self.upload_img(image, None)
        chat_state = self.ask_message(text, chat_state)
        answer, chat_state, img_list = self.answer_question(chat_state, img_list, 1, 1)
        return answer

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

    def get_image_tensor(self, image):
        image_tensor = self.chat.vis_processor(image).unsqueeze(0).to(self.chat.device)
        return image_tensor

    def generate_batch_func(self, images, texts, batch_size, max_new_tokens=512, num_beams=3):
        answers = []
        convs = [CONV_VISION.copy() for _ in range(batch_size)]
        [self.chat.ask('<Img><ImageHere></Img> {} '.format(text), conv) for conv, text in zip(convs, texts)]
        [conv.append_message(conv.roles[1], None) for conv in convs]
        # [conv.append_message(None, None) for conv in convs]
        
        with torch.no_grad():
            image_embs, _ = self.chat.model.encode_img(images.to(self.chat.device))
        image_lists = [[image_emb[None]] for image_emb in image_embs]
        
        batch_embs = [self.chat.get_context_emb(conv, img_list) for conv, img_list in zip(convs, image_lists)]    
        
        batch_size = len(batch_embs)
        max_len = max([emb.shape[1] for emb in batch_embs])
        emb_dim = batch_embs[0].shape[2]
        dtype = batch_embs[0].dtype
        device = batch_embs[0].device
        
        embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
        attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
        for i, emb in enumerate(batch_embs):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]
            attn_mask[i, -emb_len:] = 1
            
    #     outputs = self.chat.emb_generate(embs, max_new_tokens=20, attention_mask=attn_mask)
        with torch.no_grad():
            outputs = self.chat.model.llama_model.generate(
                        inputs_embeds=embs,
                        max_new_tokens=max_new_tokens,
                        attention_mask=attn_mask,
                        num_beams=num_beams,
                        do_sample=False,
                )

        for output_token in outputs:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.chat.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_texts = output_texts.split('</s>')[0]  # remove the stop sign '###'
            output_texts = output_texts.replace("<s>","")
            output_texts = output_texts.split(r'[/INST]')[-1].strip()
            answers.append(output_texts)

        return answers

    def generate_batch(self, images, texts, batch_size, max_new_tokens=10, num_beams=3):
        answers = []
        total_batches = len(texts) // batch_size

        for batch_idx in tqdm(range(total_batches), total=total_batches):
            start_idx = batch_idx * batch_size 
            end_idx = (batch_idx + 1) * batch_size 
            
            batch_texts = texts[start_idx:end_idx]
            batch_images = images[start_idx:end_idx]
            batch_answers = self.generate_batch_func(batch_images, batch_texts, batch_size, max_new_tokens, num_beams)
            answers.extend(batch_answers)
            
        
        if len(texts) % batch_size != 0:
            start_idx = total_batches * batch_size
            end_idx = len(texts)

            # Get the remaining data as a smaller batch
            batch_texts = texts[start_idx:end_idx]
            batch_images = images[start_idx:end_idx]
            batch_answers = self.generate_batch_func(batch_images, batch_texts, batch_size, max_new_tokens, num_beams)
            answers.extend(batch_answers)

        return answers