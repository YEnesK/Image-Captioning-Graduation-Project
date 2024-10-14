import json
from tqdm import tqdm
import nltk
nltk.download('wordnet')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from rouge_score import rouge_scorer

# Dosya yollarını belirleyin
dataset_test_16k_path = 'captions/dataset_test_16k.json'
model_test_result_path = 'captions/model_1k_cosmos_test_result.json'

# JSON dosyalarını yükleyin
with open(dataset_test_16k_path, 'r', encoding='utf-8') as f:
    dataset_test_16k = json.load(f)

with open(model_test_result_path, 'r', encoding='utf-8') as f:
    model_66k_test_result = json.load(f)

# BLEU4 skoru hesaplamak için fonksiyon
def calculate_bleu(reference_caption, candidate_caption):
    smoothing_function = SmoothingFunction().method4
    reference = reference_caption.split()
    candidate = candidate_caption.split()
    bleu_score = sentence_bleu([reference], candidate, smoothing_function=smoothing_function)
    return bleu_score

# BLEU3 skoru hesaplamak için fonksiyon
def calculate_bleu3(reference_caption, candidate_caption):
    reference = [reference_caption.split()]
    candidate = candidate_caption.split()
    smoothing_function = SmoothingFunction().method1
    bleu3_score = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
    return bleu3_score

# METEOR skoru hesaplamak için fonksiyon
def calculate_meteor(reference_caption, candidate_caption):
    reference = reference_caption.split()
    candidate = candidate_caption.split()
    meteor = meteor_score([reference], candidate)
    return meteor

def calculate_cider(reference_caption, candidate_caption):
    cider_scorer = Cider()
    tokenizer = PTBTokenizer()

    # Prepare data in COCO format
    gts = {0: [{'caption': reference_caption}]}
    res = {0: [{'caption': candidate_caption}]}
    
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # Calculate CIDEr score
    cider_score, _ = cider_scorer.compute_score(gts, res)
    return cider_score

def calculate_rouge_l(reference_caption, candidate_caption):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference_caption, candidate_caption)
    rouge_l_score = scores['rougeL'].fmeasure
    return rouge_l_score



total_bleu4  = 0
total_bleu3  = 0
total_meteor = 0
total_cider  = 0
total_rougel = 0

# Skorları hesaplama
for model_result in tqdm(model_66k_test_result):
    file_path = model_result['file_path']
    candidate_caption = model_result['caption'][0]

    # Aynı resim dosyasına ait referans caption'ları bul
    reference_captions = []
    for data in dataset_test_16k:
        if data['file_path'] == file_path:
            reference_captions = data['captions']
            break

    # Her bir referans cümlesi için skorlarını hesapla ve ortalamasını al
    if reference_captions:

        bleu4_scores = [calculate_bleu(ref, candidate_caption) for ref in reference_captions]
        total_bleu4  = total_bleu4 + sum(bleu4_scores) / len(bleu4_scores)

        bleu3_scores = [calculate_bleu3(ref, candidate_caption) for ref in reference_captions]
        total_bleu3  = total_bleu3 + sum(bleu3_scores) / len(bleu3_scores)

        meteor_scores = [calculate_meteor(ref, candidate_caption) for ref in reference_captions]
        total_meteor  = total_meteor + sum(meteor_scores) / len(meteor_scores)

        """ cider_scores = [calculate_cider(ref, candidate_caption) for ref in reference_captions]
        total_cider = total_cider + sum(cider_scores) / len(cider_scores) """

        rougel_scores = [calculate_rouge_l(ref, candidate_caption) for ref in reference_captions]
        total_rougel  = total_rougel + sum(rougel_scores) / len(rougel_scores)





    else:
        print(f"File: {file_path}, Reference captions not found.")

bleu4 = total_bleu4/16000
bleu3 = total_bleu3/16000
meteor = total_meteor/16000
cider = total_cider/16000
rougel = total_rougel/16000

print(model_test_result_path)

print("BLEU4: ", bleu4*100)
print("BLEU3: ", bleu3*100)
print("METEOR: ", meteor*100)
print("CIDEr: ", cider*100)
print("ROUGE-L: ", rougel*100)