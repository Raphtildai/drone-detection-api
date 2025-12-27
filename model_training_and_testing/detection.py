# detection.py

import torch

def detect_drone(model, mel_tensor, device="cpu", threshold=0.5):
    model.eval()
    mel_tensor = mel_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(mel_tensor)
        probs = torch.softmax(logits, dim=1)
    return probs[0,1].item() > threshold
