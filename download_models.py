from huggingface_hub import snapshot_download

def download_weights():
    deepSeekOCR_weights = snapshot_download("deepseek-ai/deepseek-ocr", local_dir="./models/deepseek-ocr_w", allow_patterns=["*.safetensors"])
    qwen2_5B_vl_weights = snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", local_dir="./models/qwen-2.5b-vl_w", allow_patterns=["*.safetensors"])
    print(f"Downloaded weights: {deepSeekOCR_weights}, {qwen2_5B_vl_weights}")

def download_pipe(model_name="both"):
    deepseekOCR_pipe = None
    qwen2_5B_vl_pipe = None
    
    if model_name == "both":
        deepseekOCR_pipe = snapshot_download("deepseek-ai/deepseek-ocr", local_dir="./deepseek-ocr", ignore_patterns=["*.safetensors"])
        qwen2_5B_vl_pipe = snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", local_dir="./qwen-2.5b-vl", ignore_patterns=["*.safetensors"])
    elif model_name == "qwen":
        qwen2_5B_vl_pipe = snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", local_dir="./qwen-2.5b-vl", ignore_patterns=["*.safetensors"])
    elif model_name == "deepseek":
        deepseekOCR_pipe = snapshot_download("deepseek-ai/deepseek-ocr", local_dir="./deepseek-ocr", ignore_patterns=["*.safetensors"])
    
    print(f"Downloaded pipelines: {deepseekOCR_pipe}, {qwen2_5B_vl_pipe}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download models from Hugging Face Hub")
    parser.add_argument("--download_weights", action="store_true", help="Download model weights")
    parser.add_argument("--download_pipelines", choices=["qwen", "deepseek", "both"], action="store", help="Download model pipelines")

    args = parser.parse_args()

    if args.download_weights:
        download_weights()
    if args.download_pipelines == "qwen":
        download_pipe("qwen")
    elif args.download_pipelines == "deepseek":
        download_pipe("deepseek")
    else:
        download_pipe("both")