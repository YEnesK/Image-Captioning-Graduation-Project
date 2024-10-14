# Automated Text Generation from Images Based on Deep Learning

This project aims to generate descriptive captions for images using deep learning techniques. It combines computer vision and natural language processing to analyze visual content and produce meaningful text descriptions.

## Project Overview

- **Objective**: Create a system that can analyze visual data from images, identify key objects and actions, and generate accurate and contextually relevant captions.
- **Model Architecture**: Transformer-based model using CLIP for image encoding and GPT-2 for text generation
- **Dataset**: MSCOCO (various subset sizes used for training and evaluation)
- **Language**: Turkish

## Key Components

1. **CLIP Image Encoder**: 
   - Converts input images into encoded vector arrays
   - Utilizes position embedding and multi-head attention mechanisms

2. **Vision Transformer Block (ViT)**:
   - Further processes the encoded image vectors
   - Consists of multiple layers with attention mechanisms and MLPs

3. **Text Decoder**:
   - Generates captions based on the encoded image features
   - Utilizes self-attention and cross-attention mechanisms

## Results

The project compared different configurations:
- Dataset sizes: 1k, 8k, 32k, and 66k images
- Language Models: ytu-ce-cosmos/turkish-gpt2 and redrussianarmy/gpt2-turkish-cased

Evaluation metrics used:
- BLEU_3, BLEU_4
- METEOR
- ROUGE-L
- CIDEr

Key findings:
- Larger dataset sizes generally improved performance across all metrics
- Performance gains diminished between 32k and 66k dataset sizes
- No clear advantage between the two GPT-2 models tested

## Future Work

- Explore larger dataset sizes and their impact on performance
- Investigate other pre-trained models for image encoding and text generation
- Implement techniques to improve caption quality and relevance
- Extend the system to support multiple languages

## Requirements

- Python 3.7+
- PyTorch
- Transformers library
- CLIP
- GPT-2
- NumPy
- Pandas

## Usage

(Include instructions on how to set up the environment, prepare the data, train the model, and generate captions for new images)

## Contributors

- Yusuf Enes KURT
- Muhammed Ali LALE

## Acknowledgements

This project was completed as part of the Computer Engineering program at Yildiz Technical University, under the supervision of Prof. Dr. Banu DİRİ.


