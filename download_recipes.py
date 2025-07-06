from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_unzip():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        "shuyangli94/food-com-recipes-and-user-interactions",
        path="data",
        unzip=True
    )
    print("✅ Downloaded and extracted into data/")

if __name__ == "__main__":
    download_and_unzip()
