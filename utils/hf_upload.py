from huggingface_hub import HfApi
api = HfApi(token="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")   # token from HF settings → Access Tokens
api.create_repo("pakhomovee/celeba", repo_type="dataset", exist_ok=True)
api.upload_file(
    path_or_fileobj="data/celeba.zip",
    path_in_repo="celeba.zip",
    repo_id="pakhomovee/celeba",
    repo_type="dataset",
)
