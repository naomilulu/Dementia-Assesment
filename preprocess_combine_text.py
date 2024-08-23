import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import nltk
import jieba
nltk.download('punkt')
from sklearn.metrics.pairwise import cosine_similarity

# 加載BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')
model = BertModel.from_pretrained('google-bert/bert-base-chinese')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def remove_overlap(text1, text2):
    # 使用jieba分詞
    words1 = list(jieba.cut(text1))
    words2 = list(jieba.cut(text2))

    # 找出重疊區域
    for i in range(1, len(words2)):
        overlap_part = ''.join(words2[:i])
        embedding1 = get_bert_embedding(text1)
        embedding2 = get_bert_embedding(overlap_part)
        similarity = cosine_similarity([embedding1.numpy()], [embedding2.numpy()])[0][0]

        # 設定相似度閾值，來識別重疊部分
        if similarity > 0.8:  # 可根據實際情況調整閾值
            non_overlap_part = ''.join(words2[i:])
            return non_overlap_part

    return text2


data = pd.read_csv("data.csv")
biclass = pd.read_csv("data/biclass2.txt", sep=" ")
trans = pd.read_csv("data_process/CTTsegment_diarization_overlap/transcription-v2.csv")

biclass["file"] = biclass["file"].str.replace("CTT", "")
# trans = trans.drop(188)
trans["audio"] = trans["audio"].str.split("C").str.get(0)
tran = ""
all_tran = []

for index, row in data.iterrows():
    if not(pd.isna((row["編號"])) or pd.isna((row["性別"])) or pd.isna((row["出生年月日"]))):
        df1 = biclass.loc[biclass["file"] == str(int(row["編號"]))]
        if not(df1.empty):
            df2 = trans.loc[trans["audio"] == str(int(row["編號"]))]

            tran = []
            for index1, row1 in df2.iterrows():
                tran.append(row1["transcription"])

            # print(tran)

            # 處理並合併轉錄文本
            processed_transcripts = [tran[0]]

            for i in range(1, len(tran)):
                non_overlap_text = remove_overlap(tran[i-1], tran[i])
                processed_transcripts.append(non_overlap_text)

            final_transcript = " ".join(processed_transcripts)
            # print(final_transcript)
            
            file = str(int(row["編號"]))
            ad = df1.iloc[0]["label"]
            gender = row["性別"]
            age = str(1140000-int(row["出生年月日"]))[0:2]
            
            print("年齡：", age, "性別：", row["性別"], "，")
            print("編號：", file, "，圖片描述任務文字：", final_transcript)
            
            all_tran.append({"file": file, "AD": ad, "gender": gender, "age": age, "transcription": final_transcript})

all_tran = pd.DataFrame(all_tran)
all_tran.to_csv("data_process/whisper-v2-all-diarization-overlap.csv", index=False)