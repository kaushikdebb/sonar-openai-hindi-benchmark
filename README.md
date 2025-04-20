# SONAR vs OpenAI Embeddings: Multilingual Performance Comparison

This project compares the performance of Meta's SONAR embeddings against OpenAI's text-embedding-3-large model in a multilingual context, specifically focusing on English-Hindi sentence translations.

## 🔍 Project Overview

This experiment was inspired by Meta's Large Concept Model (LCM) research, which heavily relies on SONAR embeddings. The goal was to evaluate how well SONAR performs in capturing semantic similarity across languages compared to OpenAI's embeddings.

### Key Findings
- SONAR achieved 79% average similarity compared to OpenAI's 57%
- SONAR showed 64% perfect matches (>0.8 similarity) while OpenAI had 0%
- SONAR outperformed OpenAI in 95% of the test cases

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA-capable GPU (recommended)
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kaushikdebb/sonar-openai-hindi-benchmark.git
cd sonar-vs-openai
```


2. Install required packages:
```bash
pip install torch transformers openai scikit-learn pandas numpy matplotlib
```

3. Set up OpenAI API key:
   
 Replace <OPEN_AI_KEY> in the script with your OpenAI API key
 

## 📁 Project Structure

``` bash
├── data/
│   ├── eng_hindi_translation.csv        # Original dataset
│   └── eng_hindi_translation_with_emb_sim.csv  # Results with embeddings
├── sonar_vs_openAI_embedding.py  # Main script
└── README.md
```

## 💻 Usage
Run the main script:
``` bash
python sonar_vs_openAI_embedding.py
```

The script will:

1. Load SONAR and OpenAI embedding models
2. Process the English-Hindi sentence pairs
3. Calculate cosine similarities
4. Generate comparative visualizations
5. Save results to CSV

## 📊 Visualization
The script generates four main visualizations:

1. Mean cosine similarity with standard deviation
2. Distribution boxplots comparing both models
3. Scatter plot showing direct comparison
4. Grouped bar chart by complexity level

## 🔧 Methodology

### Embedding Generation

1. SONAR: Uses cointegrated/SONAR_200_text_encoder with language-specific tokenization
2. OpenAI: Uses text-embedding-3-large API

### Comparison Metrics

1. Cosine similarity between English and Hindi pairs
2. Win rate (percentage where SONAR > OpenAI)
3. Threshold-based categorization (High: ≥0.8, Medium: 0.6-0.8, Low: <0.6)
4. Stratified analysis by sentence complexity

## 📈 Results Summary

``` bash 
mean_openai: 0.678
mean_sonar:  0.741
win_rate:    67.0% (SONAR > OpenAI)
```

## 🤝 Contributing
Feel free to open issues or submit pull requests if you have suggestions for improvement.

## 📝 Citation
If you use this code or findings in your research, please cite:

``` bash
@misc{sonar_embeddings_comparison,
  title={sonar-openai-hindi-bentchmark: Multilingual Performance Comparison},
  author={[Kaushik Deb]},
  year={2025},
  url={https://github.com/kaushikdebb/sonar-openai-hindi-bentchmark.git}
}
```

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
1. Meta AI for developing SONAR
2. OpenAI for their embedding API






