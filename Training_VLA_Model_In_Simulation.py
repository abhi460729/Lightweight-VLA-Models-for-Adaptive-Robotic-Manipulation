import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

dataset = LeRobotDataset("outputs/vla_red_cup")
model = VLAModel(action_dim=7)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    for batch in dataset.get_dataloader(batch_size=32):
        images = batch["observation.image"]  # [batch, time, C, H, W]
        text_inputs = batch["observation.text"]  # Tokenized instructions
        actions = batch["action"]  # [batch, time, action_dim]
        predicted_actions = model(images, text_inputs)
        loss = torch.mean((predicted_actions - actions) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
torch.save(model.state_dict(), "outputs/vla_red_cup_model/model.pth")
