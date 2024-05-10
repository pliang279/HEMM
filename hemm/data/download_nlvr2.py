import json 
import requests
from PIL import Image
from tqdm import tqdm
from io import BytesIO


fn = "/work/agoindan/.cache/nlvr/nlvr2/data/test1.json"
with open(fn) as f:
    data = f.readlines()

for line in data:
    dt = json.loads(line)
    idf = dt["identifier"]
    x = idf.strip().split("-")[-1]
    init = "-".join(idf.split("-")[:-1])
    print(init, x)
    img_path = f"/work/agoindan/.cache/nlvr/nlvr2/util/test1_images/{init}-img{x}*.png"
    # print(len(fls))
    # break


