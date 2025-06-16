import torch
from lerobot.common.robot_devices.robots import SO100Robot
from transformers import DistilBertTokenizer

model = VLAModel(action_dim=7)
model.load_state_dict(torch.load("outputs/vla_red_cup_model/finetuned.pth"))
model.eval()
robot = SO100Robot(fps=30, camera_type="opencv")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

instructions = ["Pick up the blue cup", "Pick up the green mug"]
results = {inst: {"success": 0, "episodes": 50} for inst in instructions}
for instruction in instructions:
    text_inputs = tokenizer(instruction, return_tensors="pt")
    for episode in range(50):
        robot.reset()
        done = False
        while not done:
            image = robot.get_image()
            state = robot.get_state()
            image_tensor = torch.tensor(image).unsqueeze(0)
            with torch.no_grad():
                action = model(image_tensor, text_inputs)
            done, success = robot.step(action.squeeze(0).numpy())
            if success:
                results[instruction]["success"] += 1
    print(f"{instruction}: Success rate {results[instruction]['success'] / 50 * 100}%")
