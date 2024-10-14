import json
import os
import torch
from model import ClipCaptionModel
from transformers import AutoTokenizer
import clip
from PIL import Image
import skimage.io as io
from utils import generate2, generate_beam
import sys

def main():
    if len(sys.argv) < 2:
        print("Lütfen bir model ismi girin!")
        return
    
    model = sys.argv[1]

    # Model ve diğer parametrelerin ayarlanması
    prefix_length = 10
    prefix_dim = 512

    ai_model = ClipCaptionModel(prefix_length, prefix_size=prefix_dim)
    ai_model.load_state_dict(torch.load("checkpoints/model_"+model+".pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ai_model = ai_model.to(device)
    clip_model, clip_preprocessor = clip.load("ViT-B/32", device=device)
    use_beam_search = True
    # tokenizer = AutoTokenizer.from_pretrained("redrussianarmy/gpt2-turkish-cased")
    tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-base-bert-uncased")

    dataname = 'dataset_test_16k.json'

    # dataset_test_16k.json dosyasını okuma
    with open(dataname, 'r') as f:
        dataset = json.load(f)

    results = []

    sayac = 0

    # Tüm resimler için metin üretme
    for item in dataset:

        if(sayac%160==0):
            print(sayac/160)

        sayac = sayac + 1

        file_path = item["file_path"]
        image_path = os.path.join("images_data", file_path)
        image = io.imread(image_path)

        pil_image = Image.fromarray(image)
        image = clip_preprocessor(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
            prefix_embed = ai_model.clip_project(prefix).reshape(1, prefix_length, -1)

        if use_beam_search:
            generated_text_prefix = generate_beam(ai_model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(ai_model, tokenizer, embed=prefix_embed)

        results.append({
            "caption": [generated_text_prefix],
            "file_path": file_path
        })

    results_filename='model_'+model+'_test_result.json'
    # Sonuçları model_16k_test_result.json dosyasına kaydetme
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Metin üretimi tamamlandı ve sonuçlar "+results_filename+" dosyasına kaydedildi.")

if __name__ == "__main__":
    main()
