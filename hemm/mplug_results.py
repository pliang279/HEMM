import pickle 

data = pickle.load(open("/work/agoindan/.cache/mplugowl_vqa.pkl", "rb"))
gts = pickle.load(open("/work/agoindan/.cache/vqa.pkl", "rb"))

print(len(data["predictions"]), gts["texts"])
# for i in range(len(data["predictions"])):
#     pred = data["predictions"][i]
#     gt = data["gts"][i]
#     print("*"*100)
#     print(pred)
#     print("#"*100)
#     print(gt)

