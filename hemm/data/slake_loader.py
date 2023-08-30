from slake_dataset import SlakeVQA
from torch.utils.data import DataLoader
from lavis.models import load_model_and_preprocess

model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
)
dataset = SlakeVQA("./drive/MyDrive/Slake1.0/imgs", vis_processors,"./drive/MyDrive/Slake1.0/test.json", "cuda")
loader = DataLoader(dataset, batch_size=1)

imgs = []
prompts = []
gts = []
batch_size = 16

for _, sample in enumerate(loader):
    img, prompt, gt = sample["image"], sample["prompt"], sample["gt"]
    imgs.append(img)
    prompts.append(prompt[0])
    gts.append(gt[0])

for i in tqdm(range(0, len(imgs), batch_size)):
    img_batch = imgs[i : i + batch_size]
    prompt_batch = prompts[i : i + batch_size]
    gts_batch = gt[i: i + batch_size]
    img_batch = torch.cat(img_batch)
    pred = model.generate({"image": img_batch, "prompt": prompt_batch})
