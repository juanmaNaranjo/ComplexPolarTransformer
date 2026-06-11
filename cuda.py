import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda built:", torch.backends.cuda.is_built())
print("torch cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu 0:", torch.cuda.get_device_name(0))