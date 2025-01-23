import kagglehub

# Download latest version
path = kagglehub.dataset_download("imsparsh/lakh-midi-clean")

print("Path to dataset files:", path)
