import os
import subprocess
import argparse
import re

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
    args = parser.parse_args(["--cfg-path", "hemm/models/minigpt4/configs/eval_configs/minigpt4_eval.yaml"])
    return args

class MiniGPT4(HEMMModel):
    def __init__(self):
        pass
    
    def load_weights(self):
        # if not os.path.exists('MiniGPT-4'):
        #     subprocess.Popen('git clone https://github.com/Vision-CAIR/MiniGPT-4.git', shell=True)
        shell_command('wget https://huggingface.co/wangrongsheng/MiniGPT4-7B/resolve/main/prerained_minigpt4_7b.pth')

        # Read in the file
        with open('hemm/models/minigpt4/configs/models/minigpt4.yaml', 'r') as file:
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace('/path/to/vicuna/weights/', 'wangrongsheng/MiniGPT-4-LLaMA-7B')

        # Write the file out again
        with open('hemm/models/minigpt4/configs/models/minigpt4.yaml', 'w') as file:
            file.write(filedata)

        # Read in the file
        with open('hemm/models/minigpt4/configs/eval_configs/minigpt4_eval.yaml', 'r') as file:
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace('/path/to/pretrained/ckpt/', 'prerained_minigpt4_7b.pth')

        # Write the file out again
        with open('hemm/models/minigpt4/configs/eval_configs/minigpt4_eval.yaml', 'w') as file:
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
        elif dataset_key == 'memotion' or dataset_key == 'scienceqa' or dataset_key == 'vcr':
            match = re.search(r"\b\d\b", text)
            if match:
                first_number = int(match.group())
                return first_number
            else:
                return None