import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torch.optim import AdamW

dataset = LeRobotDataset("outputs/vla_red_cup_real")
model = VLAModel(action_dim=7)
model.load_state_dict(torch.load("outputs/vla_red_cup_model/model.pth"))
optimizer = AdamW(model.parameters(), lr=1e-5)  # Lower learning rate

for epoch in range(10):
    for batch in dataset.get_dataloader(batch_size=16):
        images = batch["observation.image"]
        text_inputs = batch["observation.text"]
        actions = batch["action"]
        predicted_actions = model(images, text_inputs)
        loss = torch.mean((predicted_actions - actions) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
torch.save(model.state_dict(), "outputs/vla_red_cup_model/finetuned.pth")
