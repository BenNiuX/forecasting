import os
import requests
import base64

def gen_img_bria(prompt):
    # https://build.nvidia.com/briaai/bria-2_3
    # https://docs.api.nvidia.com/nim/reference/briaai-bria-2_3-infer
    invoke_url = os.getenv("IMG_GEN_URL_BRIA")
    prompt = "Create an image to visualise the following information: " + prompt
    headers = {
        "Authorization": "Bearer " + os.getenv("NIM_API_KEY"),
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": prompt,
        "negative_prompt": "",
        "mode": "text-to-image",
        "model": "bria-2.3",
        "output_format": "jpeg",
        "aspect_ratio": "16:9",
        "seed": 0,
        "cfg_scale": 5,
        "steps": 30
    }

    response = requests.post(invoke_url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    seed = data['seed']
    print("Seed", data['seed'], "Finish Reason", data['finish_reason']) 
    img_base64 = data['image']
    # imageBytes = base64.b64decode(data['image'])
    # with open(f'bria_{seed}.jpg', 'wb') as f:
    #     f.write(imageBytes)
    return img_base64

def gen_img_consi_story():
    # https://build.nvidia.com/nvidia/consistory
    # https://docs.api.nvidia.com/nim/reference/nvidia-consistory
    invoke_url = os.getenv("IMG_GEN_URL_CONSI")
    prompt = "Social impact to China: Nationalistic sentiments could be heightened, but potential for domestic unrest if the invasion leads to economic or political fallout."
    #"an old woman wearing a dress"
    subject_words = ["social", "China", "invasion"]#["woman", "dress"]
    prompt_sce1 = "Nationalistic sentiments could be heightened"#"walking in the garden"
    prompt_sce2 = "potential for domestic unrest"#"feeding birds in the square"
    headers = {
        "Authorization": "Bearer " + os.getenv("NIM_API_KEY"),
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    payload = {
        "mode": 'init',
        "subject_prompt": prompt,
        "subject_tokens": subject_words,
        "subject_seed": 0,
        "style_prompt": "A photo of",
        "scene_prompt1": prompt_sce1,
        "scene_prompt2": prompt_sce2,
        "negative_prompt": "",
        "cfg_scale": 5,
        "same_initial_noise": False
    }

    response = requests.post(invoke_url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()

    for idx, img_data in enumerate(data['artifacts']):
        seed = img_data["seed"]
        print("Seed", seed, "Finish Reason", img_data["finishReason"]) 
        img_base64 = img_data["base64"]
        img_bytes = base64.b64decode(img_base64)
        with open(f'consi_{idx}_{seed}.jpg', "wb") as f:
            f.write(img_bytes)
