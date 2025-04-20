# sonar-openai-hindi-benchmark


SONAR vs OpenAI Embeddings: Multilingual Performance Comparison
This project compares the performance of Meta's SONAR embeddings against OpenAI's text-embedding-3-large model in a multilingual context, specifically focusing on English-Hindi sentence translations.
🔍 Project Overview
This experiment was inspired by Meta's Large Concept Model (LCM) research, which heavily relies on SONAR embeddings. The goal was to evaluate how well SONAR performs in capturing semantic similarity across languages compared to OpenAI's embeddings.
Key Findings

SONAR achieved 79% average similarity compared to OpenAI's 57%
SONAR showed 64% perfect matches (>0.8 similarity) while OpenAI had 0%
SONAR outperformed OpenAI in 95% of the test cases

🚀 Getting Started
Prerequisites

Python 3.8+
PyTorch
CUDA-capable GPU (recommended)
OpenAI API key

Installation

Clone the repository:

bashgit clone https://github.com/your-username/sonar-vs-openai.git
cd sonar-vs-openai

Install required packages:

bashpip install torch transformers openai scikit-learn pandas numpy matplotlib

Set up OpenAI API key:


Replace <OPEN_AI_KEY> in the script with your OpenAI API key

📁 Project Structure
├── data/
│   ├── eng_hindi_translation.csv        # Original dataset
│   └── eng_hindi_translation_with_emb_sim.csv  # Results with embeddings
├── sonar_vs_openAI_embedding.py  # Main script
└── README.md
💻 Usage
Run the main script:
bashpython sonar_vs_openAI_embedding.py
The script will:

Load SONAR and OpenAI embedding models
Process the English-Hindi sentence pairs
Calculate cosine similarities
Generate comparative visualizations
Save results to CSV

📊 Visualization
The script generates four main visualizations:

Mean cosine similarity with standard deviation
Distribution boxplots comparing both models
Scatter plot showing direct comparison
Grouped bar chart by complexity level

🔧 Methodology
Embedding Generation

SONAR: Uses cointegrated/SONAR_200_text_encoder with language-specific tokenization
OpenAI: Uses text-embedding-3-large API

Comparison Metrics

Cosine similarity between English and Hindi pairs
Win rate (percentage where SONAR > OpenAI)
Threshold-based categorization (High: ≥0.8, Medium: 0.6-0.8, Low: <0.6)
Stratified analysis by sentence complexity

📈 Results Summary
MetricOpenAISONARMean Similarity0.570.79Median Similarity0.590.82High-Quality Matches (≥0.8)0%64%
🤝 Contributing
Feel free to open issues or submit pull requests if you have suggestions for improvement.
📝 Citation
If you use this code or findings in your research, please cite:
@misc{sonar_embeddings_comparison,
  title={SONAR vs OpenAI Embeddings: Multilingual Performance Comparison},
  author={[Your Name]},
  year={2025},
  url={https://github.com/your-username/sonar-vs-openai}
}
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Meta AI for developing SONAR
OpenAI for their embedding API
The creators of the English-Hindi translation dataset
