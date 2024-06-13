dataset_categories = {
    "use_case": {
        "multimedia": ["vqa", "mmimdb", "visualgen", "okvqa", "flickr30k", "vcr", "nocaps", "nlvr", "GQA", "nlvr2", "irfl", "winoground"],
        "affect": ["newyorkercartoon", "hatefulmemes", "memecaps", "memotion", "faceemotion"],
        "science": ["scienceqa", "decimer", "inat", "ucmerced", "resisc45"],
        "health": ["pathvqa", "slake", "vqarad", "plip"],
        "hci": ["enrico", "screen2words"],
    },
    "granularity": {
        "fine_grained": ["nlvr", "inat", "GQA", "winoground", "vqarad", "vqa", "visualgen",
                         "vcr", "slake", "plip", "pathvqa", "okvqa"],
        "not_fine_grained": ["nlvr2", "newyorkercartoon", "mmimdb", "memotion", "memecaps", 
                             "irfl", "hatefulmemes", "flickr30k", "enrico", "faceemotion",
                             "decimer", "ucmerced", "screen2words", "scienceqa", "resisc45",
                             "nocaps"]
    },
    "reasoning": {
        "less_reasoning": ["nlvr2", "nlvr", "mmimdb", "inat", "flickr30k", "GQA", 
                            "enrico", "faceemotion", "winoground", "vqa", "visualgen",
                            "vcr", "ucmerced", "screen2words", "scienceqa", "resisc45",
                            "pathvqa", "nocaps", "okvqa"],
        "more_reasoning": ["newyorkercartoon", "memotion", "memecaps", "irfl", "hatefulmemes",
                            "decimer", "vqarad", "slake", "plip"],
    },
    "knowledge": {
        "external_knowledge": ["newyorkercartoon", "memotion", "memecaps", "hatefulmemes", "inat",
                                "decimer", "vqarad", "slake", "scienceqa", "plip", "pathvqa", "okvqa"],
        "no_external_knowledge": ["nlvr", "nlvr2", "mmimdb", "irfl", "flickr30k",
                                    "GQA", "enrico", "faceemotion", "winoground", "vqa",
                                    "visualgen", "vcr", "ucmerced", "screen2words", "resisc45",
                                    "nocaps"],
    },
    "interactions": {
        "redundancy": ["nlvr", "nlvr2", "GQA", "winoground", "vqarad", "vqa", "visualgen",
                       "vcr", "slake", "plip", "pathvqa", "okvqa"],
        "uniqueness": ["inat", "flickr30k", "enrico", "faceemotion", "decimer", "ucmerced",
                       "screen2words", "resisc45", "nocaps"], 
        "synergy": ["newyorkercartoon", "mmimdb", "memotion", "memecaps", "irfl", "hatefulmemes",
                    "scienceqa"],
    },
    "information_flow": {
        "querying": ["nlvr2", "nlvr", "inat", "GQA", "enrico", "faceemotion", "winoground",
                    "vqarad", "vqa", "visualgen", "ucmerced", "slake", "resisc45", "plip",
                    "pathvqa", "okvqa"], 
        "translation": ["flickr30k", "decimer", "screen2words", "nocaps"], 
        "fusion": ["newyorkercartoon", "mmimdb", "memotion", "memecaps", "irfl", "hatefulmemes",
                   "vcr", "scienceqa"],
    },
}

model_categories = {
    "diversity": {
        "diverse": ["instruct_blip", "emu", "fuyu", "gptv", "mplugowl", "gemini", "kosmos2"],
        "non_diverse": ["llama_adapter", "openflamingo", "blip2", "minigpt4"],
    },
    "tuning": {
        "instruction_tuning": ["minigpt4", "instruct_blip", "kosmos2", "llama_adapter", "gptv", "mplugowl", "gemini"],
        "supervised_fine_tuning": ["blip2", "openflamingo", "emu", "fuyu"],
    },
    "modalities": {
        "interleaved": ["emu", "openflamingo", "gemini", "kosmos2", "fuyu"],
        "separate": ["blip2", "instruct_blip", "minigpt4", "llama_adapter", "mplugowl"],
    },
    "training_type": {
        "end_to_end": ["emu", "fuyu", "kosmos2"],
        "modular_fine_tune": ["blip2", "instruct_blip", "minigpt4", "openflamingo", "llama_adapter", "mplugowl"],
    },
    "pre_training_data_size": {
        "small": ["minigpt4", "llama_adapter", "emu"],
        "medium": ["blip2", "instruct_blip", "openflamingo", "fuyu"],
        "large": ["gemini", "gptv", "mplugowl", "kosmos2"],
    },
    "trainable_params": {
        "small": ["blip2", "instruct_blip", "minigpt4", "llama_adapter"],
        "medium": ["emu", "fuyu", "mplugowl", "openflamingo", "kosmos2"],
        "large": ["gptv", "gemini"],
    },
    "total_params": {
        "small": ["instruct_blip", "openflamingo", "kosmos2"],
        "medium": ["blip2", "minigpt4", "emu", "llama_adapter", "fuyu", "mplugowl"],
        "large": ["gptv", "gemini"],
    }
}

total_number_of_params = {
    "blip2": 12.1,
    "instruct_blip": 4,
    "minigpt4": 13, 
    "emu": 14,
    "openflamingo": 3.2,
    "mplugowl": 7.2,
    "kosmos2": 1.6,
    "fuyu": 9.3,
    "gemini": 600,
    "gptv": 600, 
    "llama_adapter": 7,
}

model_to_name = {
    "minigpt4": "MiniGPT-4",
    "fuyu": "Fuyu-8B",
    "emu": "Emu",
    "gptv": "GPT-4V (estimated)",
    "gemini": "Gemini 1.0 Pro Vision (estimated)",
    "openflamingo": "OpenFlamingo-3B",
    "blip2": "BLIP-2",
    "instruct_blip": "InstructBLIP",
    "kosmos2": "Kosmos-2",
    "mplugowl": "mPLUG-Owl",
    "llama_adapter": "LLaMA-Adapter V2"
}

dset_to_name = {
    "hatefulmemes": "Hateful Memes",
    "memotion": "Memotion",
    "winoground": "Winoground",
    "faceemotion": "Face Emotion",
    "vqa": "VQA v1",
    "okvqa": "OK-VQA",
    "nlvr": "NLVR",
    "nlvr2": "NLVR2",
    "decimer": "Decimer",
    "enrico": "Enrico",
    "screen2words": "Screen2Words",
    "GQA": "GQA",
    "visualgen": "VisualGenome",
    "memecaps": "MemeCap",
    "scienceqa": "ScienceQA",
    "mmimdb": "MM-IMDb",
    "nocaps": "Nocaps",
    "slake": "SLAKE", 
    "pathvqa": "PathVQA",
    "ucmerced": "UC Merced",
    "resisc45": "RESISC45",
    "vqarad": "VQA-RAD",
    "flickr30k": "Flickr30k",
    "newyorkercartoon": "New Yorker Cartoon",
    "inat": "iNaturalist",
    "vcr": "VCR",
    "plip": "OpenPath",
    "irfl": "IRFL",
}
