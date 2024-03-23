import os
import zipfile
import pandas as pd

def check_dataset():
    folder_name = "utils/dataset/ALT-Parallel-Corpus-20191206"
    zip_file_path = "utils/dataset/ALT-Parallel-Corpus-20191206.zip"
    if not os.path.exists(folder_name):
        print("Downloading dataset...")
        os.system("wget https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/ALT-Parallel-Corpus-20191206.zip -P utils/dataset/")
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall("utils/dataset/")
        print("Dataset extracted.")
        os.remove(zip_file_path)
        print("Zip file removed.")
    else:
        print("Dataset already exists.")

def prepare_dataset(languages):
    check_dataset()
    dataframes = []
    for lang in languages:
        if lang not in ["bg", "en", "fil", "hi", "id", "ja", "khm", "lo", "ms", "my", "the", "vi", "zh"]:
            print(f"Language {lang} not supported.")
            continue
        filename = f"utils/dataset/ALT-Parallel-Corpus-20191206/data_{lang}.txt"
        if os.path.exists(filename):
            df = pd.read_csv(filename, sep='\t', header=None, names=['index', lang])
            dataframes.append(df)
        else:
            print(f"Data file for {lang} not found.")
    if len(dataframes) == 0:
        print("No valid data found for specified languages.")
        return None
    df_merged = dataframes[0]
    for i in range(1, len(dataframes)):
        df_merged = pd.merge(df_merged, dataframes[i], on='index')
    df_merged.columns = ['index'] + languages
    return df_merged



# Example usage:
languages = ["hi", "zh", "en"]
df = prepare_dataset(languages)
# pd.set_option('display.max_colwidth', None) # print full col
pd.set_option('display.width', None) # print full row
if df is not None:
    print(df.head())
