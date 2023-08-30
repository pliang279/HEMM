from slake_dataset_batch import SlakeVQABatch
from torch.utils.data import DataLoader

dataset = SlakeVQABatch("./drive/MyDrive/Slake1.0/imgs", vis_processors,"./drive/MyDrive/Slake1.0/test.json", "cuda")
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
