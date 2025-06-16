import torch
import torch.nn as nn
from transformers import ViTModel, DistilBertModel

class VLAModel(nn.Module):
    def __init__(self, vision_model_name="google/vit-base-patch16-224", 
                 language_model_name="distilbert-base-uncased", action_dim=7):
        super(VLAModel, self).__init__()
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        self.language_encoder = DistilBertModel.from_pretrained(language_model_name)
        for param in self.language_encoder.base_model.encoder.layer[:4]:
            param.requires_grad = False  # Freeze initial layers
        self.fusion = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.action_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=768, nhead=8), num_layers=6
        )
        self.action_head = nn.Linear(768, action_dim)

    def forward(self, images, text_inputs, attention_mask=None):
        vision_features = self.vision_encoder(images).last_hidden_state
        language_features = self.language_encoder(**text_inputs).last_hidden_state
        fused_features, _ = self.fusion(vision_features, language_features, language_features)
        action_sequence = self.action_decoder(fused_features, vision_features)
        actions = self.action_head(action_sequence)
        return actions
