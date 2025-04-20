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
git clone https://github.com/your-username/sonar-vs-openai.git
cd sonar-vs-openai
