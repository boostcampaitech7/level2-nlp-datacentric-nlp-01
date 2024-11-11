import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 모델 및 토크나이저 로드를 위한 캐시 함수
@st.cache_resource
def load_model_and_tokenizer(model_path, num_labels=7):
    tokenizer = AutoTokenizer.from_pretrained("Copycats/koelectra-base-v3-generalized-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=num_labels, 
        ignore_mismatched_sizes=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, tokenizer, device

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128, device='cuda'):
        self.texts = dataframe['text'].astype(str).fillna('').tolist()
        self.labels = dataframe['target'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len, 
            return_tensors='pt'
        )
        inputs = {key: val.squeeze(0).to(self.device) for key, val in inputs.items()}
        inputs['labels'] = torch.tensor(label, dtype=torch.long).to(self.device)
        return inputs

@st.cache_data
def load_and_process_data(data_path):
    return pd.read_csv(data_path)

def visualize_clusters(df, model, tokenizer, device):
    dataset = CustomDataset(df, tokenizer, device=device)
    
    # Extract CLS token vectors
    cls_vectors = []
    labels = []

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i % 100 == 0:
                st.write(f"처리 중... {i}/{len(dataset)}")
            
            output = model(
                input_ids=sample['input_ids'].unsqueeze(0),
                attention_mask=sample['attention_mask'].unsqueeze(0),
                output_hidden_states=True
            )
            cls_vector = output.hidden_states[-1][:, 0, :].squeeze().cpu().numpy()
            cls_vectors.append(cls_vector)
            labels.append(sample['labels'].cpu().item())

    # t-SNE dimension reduction
    with st.spinner('t-SNE 차원 축소 진행 중...'):
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=40,
            n_iter=2000,
            learning_rate='auto',
            init='pca'
        )
        cls_vectors_2d = tsne.fit_transform(np.array(cls_vectors))

    # Create dataframe for plotting
    df_plot = pd.DataFrame(cls_vectors_2d, columns=['x', 'y'])
    df_plot['target'] = labels

    # Calculate class counts
    class_counts = df_plot['target'].value_counts().sort_index()
    
    # Visualization
    fig = plt.figure(figsize=(15, 12))
    
    scatter = sns.scatterplot(
        data=df_plot,
        x='x',
        y='y',
        hue='target',
        palette='deep',
        s=120,
        alpha=0.7,
        legend='full'
    )

    for target in df_plot['target'].unique():
        mask = df_plot['target'] == target
        center_x = df_plot[mask]['x'].mean()
        center_y = df_plot[mask]['y'].mean()
        plt.scatter(
            center_x, 
            center_y, 
            c='red', 
            marker='*', 
            s=200, 
            label=f'Center {target}',
            alpha=0.7
        )

    target_names = {
        i: f"({class_counts[i]})" for i in range(7)
    }
    
    handles, labels = scatter.get_legend_handles_labels()
    labels = [target_names.get(int(label), label) for label in labels if label.isdigit()]
    plt.legend(
        handles[:7], 
        labels,
        title="category(0-6)",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0,
        fontsize=10
    )

    plt.xlabel("t-SNE Dimension 1", fontsize=12, fontweight='bold')
    plt.ylabel("t-SNE Dimension 2", fontsize=12, fontweight='bold')
    plt.title(
        "CLS Token Embedding distribution visualization \n(t-SNE)", 
        pad=20, 
        fontsize=16, 
        fontweight='bold'
    )

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.box(True)
    plt.tight_layout()
    
    return fig

def main():
    st.title("임베딩 시각화")
    
    # 사이드바에 파일 업로드 추가
    st.sidebar.header("설정")
    uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=['csv'])
    model_path = st.sidebar.text_input(
        "모델 경로를 입력하세요",
        value="/data/ephemeral/home/hsk/level2-nlp-datacentric-nlp-01/src/EDA/results/checkpoint-525"
    )
    
    if uploaded_file is not None and model_path:
        try:
            # 모델 로드 (캐시 사용)
            model, tokenizer, device = load_model_and_tokenizer(model_path)
            st.success("모델 로드 완료!")
            
            # 데이터 로드 (캐시 사용)
            df = load_and_process_data(uploaded_file)
            st.success("데이터 로드 완료!")
            
            # 시각화 버튼
            if st.button("시각화 시작"):
                with st.spinner('시각화 생성 중...'):
                    fig = visualize_clusters(df, model, tokenizer, device)
                    st.pyplot(fig)
                st.success("시각화 완료!")
                
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()