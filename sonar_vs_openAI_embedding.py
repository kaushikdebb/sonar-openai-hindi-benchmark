# Example Setup Snippets (Conceptual)
import os
import torch
import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
# high‑DPI “retina” quality
%config InlineBackend.figure_format = 'retina'

# default size for all figures
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 6)


SONAR_MODEL_NAME = "cointegrated/SONAR_200_text_encoder"
sonar_tokenizer = AutoTokenizer.from_pretrained(SONAR_MODEL_NAME)
# Load model to appropriate device (GPU highly recommended)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sonar_encoder = M2M100Encoder.from_pretrained(SONAR_MODEL_NAME).to(device).eval()



def get_sonar_embeddings(texts, lang_code):
    sonar_tokenizer.src_lang = lang_code
    inputs = sonar_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = sonar_encoder(**inputs)
        last_hidden_states = outputs.last_hidden_state
    # Mean pooling
    mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_states.size()).float()
    pooled_embeddings = torch.sum(last_hidden_states * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    # Normalize (OpenAI embeddings are normalized, good practice for comparison)
    pooled_embeddings = torch.nn.functional.normalize(pooled_embeddings, p=2, dim=1)
    return pooled_embeddings.cpu().numpy()


def create_openAI_emb(sentence1, sentence2):
    client = OpenAI(api_key=<OPEN_AI_KEY>)
    resp = client.embeddings.create(
        input=[sentence1, sentence2],
        model="text-embedding-3-large"
    )
    emb1 = np.array(resp.data[0].embedding)
    emb2 = np.array(resp.data[1].embedding)

    return emb1, emb2


def cosine_similairy_np(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def calculate_metrics(openai_sims: np.ndarray, sonar_sims: np.ndarray, labels: list[str]) -> dict:
    """Compute summary metrics for OpenAI vs SONAR similarity arrays."""
    def mean_cos(sim): return float(np.mean(sim))
    def median_iqr(sim):
        q1, q3 = np.percentile(sim, 25), np.percentile(sim, 75)
        return float(np.median(sim)), float(q3 - q1)
    def avg_delta(o, s): return float(np.mean(s - o))
    def win_rate(o, s): return float(np.sum(s > o) / len(o) * 100)
    def thresh_counts(sim, high=0.8, mid=0.6):
        return {
            f'≥{high}': int(np.sum(sim >= high)),
            f'{mid}–{high}': int(np.sum((sim >= mid) & (sim < high))),
            f'<{mid}': int(np.sum(sim < mid))
        }
    def percentiles(sim, ps=[10,25,50,75,90]):
        return {f'{p}th': float(np.percentile(sim, p)) for p in ps}
    
    strat_df = (
        pd.DataFrame({'OpenAI': openai_sims, 'SONAR': sonar_sims, 'Complexity': labels})
          .groupby('Complexity', sort=False)
          .agg(mean_openai=('OpenAI','mean'),
               mean_sonar=('SONAR','mean'))
          .reindex(["Low","Medium","High"])
          .reset_index()
    )
    
    return {
        'mean_openai': mean_cos(openai_sims),
        'mean_sonar':  mean_cos(sonar_sims),
        'median_openai': median_iqr(openai_sims)[0],
        'iqr_openai':    median_iqr(openai_sims)[1],
        'median_sonar':  median_iqr(sonar_sims)[0],
        'iqr_sonar':     median_iqr(sonar_sims)[1],
        'avg_delta':     avg_delta(openai_sims, sonar_sims),
        'win_rate':      win_rate(openai_sims, sonar_sims),
        'threshold_openai': thresh_counts(openai_sims),
        'threshold_sonar':  thresh_counts(sonar_sims),
        'percentiles_openai': percentiles(openai_sims),
        'percentiles_sonar':  percentiles(sonar_sims),
        'stratified_means':   strat_df
    }

def plot_visualizations(openai_sims: np.ndarray, sonar_sims: np.ndarray, labels: list[str],cmap='tab10'):
    """Generate four styled plots comparing OpenAI vs SONAR similarities."""
    # Use a clean, modern style
    plt.style.use('ggplot')
    cmap = plt.get_cmap(cmap)
    c_openai, c_sonar = cmap(0), cmap(1)
    
    models = ['OpenAI', 'SONAR']
    
    # 1. Mean ± Std‑Dev bar chart
    plt.figure()
    x = np.arange(2)
    means = [openai_sims.mean(), sonar_sims.mean()]
    stds  = [openai_sims.std(),  sonar_sims.std()]
    plt.bar(x, means, yerr=stds, color=[c_openai, c_sonar], capsize=8, edgecolor='k')
    plt.xticks(x, models)
    plt.ylabel("Cosine Similarity")
    plt.title("Mean Cosine Similarity ± Std Dev")
    plt.tight_layout()
    plt.show()
    
    # 2. Boxplot of distributions
    plt.figure()
    bp = plt.boxplot([openai_sims, sonar_sims],
                     labels=models,
                     patch_artist=True,
                     boxprops=dict(color='k'),
                     medianprops=dict(color='yellow'))
    bp['boxes'][0].set_facecolor(c_openai)
    bp['boxes'][1].set_facecolor(c_sonar)
    plt.ylabel("Cosine Similarity")
    plt.title("Distribution of Cosine Similarities")
    plt.tight_layout()
    plt.show()
    
    # 3. Scatter plot with 45° reference line
    plt.figure()
    plt.scatter(openai_sims, sonar_sims, color=c_sonar, alpha=0.7, edgecolor='w', s=80)
    lim = max(openai_sims.max(), sonar_sims.max())
    plt.plot([0,lim], [0,lim], color='gray', linestyle='--')
    plt.xlabel("OpenAI Similarity")
    plt.ylabel("SONAR Similarity")
    plt.title("OpenAI vs SONAR Similarity")
    plt.tight_layout()
    plt.show()
    
    # 4. Grouped bar chart by complexity
    df = pd.DataFrame({'OpenAI': openai_sims, 'SONAR': sonar_sims, 'Complexity': labels})
    grouped = df.groupby("Complexity").mean().reindex(["Low","Medium","High"])
    plt.figure()
    idx = np.arange(len(grouped))
    width = 0.35
    plt.bar(idx - width/2, grouped["OpenAI"], width, label='OpenAI', color=c_openai, edgecolor='k')
    plt.bar(idx + width/2, grouped["SONAR"], width, label='SONAR', color=c_sonar, edgecolor='k')
    plt.xticks(idx, grouped.index)
    plt.ylabel("Mean Cosine Similarity")
    plt.title("Mean Similarity by Complexity Level")
    plt.legend()
    plt.tight_layout()
    plt.show()


df_ = pd.read_csv('data/eng_hindi_translation.csv')


sonar_eng_emb = []
sonar_hin_emb = []

openai_eng_emb = []
openai_hin_emb = []

sonar_cos_sim = []
openai_cos_sim = []

for idx, row in df_.iterrows():
    sen1 = row['English']
    sen2 = row['Hindi']
    complx = row['Complexity']

    emb12 = get_sonar_embeddings([sen1], lang_code = 'eng_Latn')[0]
    emb22 = get_sonar_embeddings([sen2], lang_code = 'hin_Deva')[0]
    sonar_sim = cosine_similairy_np(emb12, emb22)
    sonar_eng_emb.append(emb12)
    sonar_hin_emb.append(emb22)
    sonar_cos_sim.append(sonar_sim)


    emb21, emb22 = create_openAI_emb(sen1, sen2)
    openai_sim = cosine_similairy_np(emb21, emb22)

    openai_eng_emb.append(emb21)
    openai_hin_emb.append(emb22)
    openai_cos_sim.append(openai_sim)

    print("English sentence = ", sen1)
    print("Hindi sentence = ", sen2)
    print("Sonar Similarity = ", sonar_sim)
    print("OpenAi Similarity = ", openai_sim)
    print("#############")
    #break
    

df_['sonar_eng_emb'] =  sonar_eng_emb
df_['sonar_hin_emb'] =  sonar_hin_emb
df_['openai_eng_emb'] =  openai_eng_emb
df_['openai_hin_emb'] =  openai_hin_emb
df_['sonar_cos_sim'] =  sonar_cos_sim
df_['openai_cos_sim'] =  openai_cos_sim

df_.to_csv('data/eng_hindi_translation_with_emb_sim.csv', index= False)

df_ = pd.read_csv('data/eng_hindi_translation_with_emb_sim.csv')
openai_sims = df_['openai_cos_sim']
sonar_sims  = df_['sonar_cos_sim']
complexity_labels = df_['Complexity']

metrics = calculate_metrics(openai_sims, sonar_sims, complexity_labels)
print("Computed metrics:")
for k, v in metrics.items():
    print(f"{k}:")
    print(v if not isinstance(v, pd.DataFrame) else v.to_string(index=False))
plot_visualizations(openai_sims, sonar_sims, complexity_labels, cmap = 'Pastel1')


