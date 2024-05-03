import numpy as np
import json
from pathlib import Path
import os
import random
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split

EPISODE_DIR = (
    "/projectnb/ds598xz/students/xthomas/FINAL/sp2024_RL/full_episodes/top_agents"
)
MODEL_DIR = "/projectnb/ds598xz/students/xthomas/FINAL/lux-AI/models/luxnet"
FINE_TUNE_MODEL = (
    "/projectnb/ds598xz/students/xthomas/FINAL/sp2024_RL/models/luxnet/top/model.pth"
)


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed = 42
seed_everything(seed)


def to_label(action):
    strs = action.split(" ")
    unit_id = strs[1]
    if strs[0] == "m":
        label = {"c": None, "n": 0, "s": 1, "w": 2, "e": 3}[strs[2]]
    elif strs[0] == "bcity":
        label = 4
    else:
        label = None
    return unit_id, label


def depleted_resources(obs):
    for u in obs["updates"]:
        if u.split(" ")[0] == "r":
            return False
    return True


def create_dataset_from_json(episode_dir, team_name="Toad Brigade"):
    obses = {}
    samples = []
    append = samples.append
    # get all .json files under the episode_dir even in subdirectories
    episodes = list(Path(episode_dir).rglob("*.json"))
    # clean episode files
    episodes = [
        str(ep) for ep in episodes if "info" not in str(ep) and "output" not in str(ep)
    ]
    for filepath in tqdm(episodes):
        with open(filepath) as f:
            json_load = json.load(f)

        ep_id = json_load["info"]["EpisodeId"]
        index = np.argmax([r or 0 for r in json_load["rewards"]])
        if json_load["info"]["TeamNames"][index] != team_name:
            continue

        for i in range(len(json_load["steps"]) - 1):
            if json_load["steps"][i][index]["status"] == "ACTIVE":
                actions = json_load["steps"][i + 1][index]["action"]
                obs = json_load["steps"][i][0]["observation"]

                if depleted_resources(obs):
                    break

                obs["player"] = index
                obs = dict(
                    [
                        (k, v)
                        for k, v in obs.items()
                        if k in ["step", "updates", "player", "width", "height"]
                    ]
                )
                obs_id = f"{ep_id}_{i}"
                obses[obs_id] = obs

                for action in actions:
                    unit_id, label = to_label(action)
                    if label is not None:
                        append((obs_id, unit_id, label))

    return obses, samples


episode_dir = EPISODE_DIR
obses, samples = create_dataset_from_json(episode_dir)
print("obses:", len(obses), "samples:", len(samples))


labels = [sample[-1] for sample in samples]
actions = ["north", "south", "west", "east", "bcity"]
for value, count in zip(*np.unique(labels, return_counts=True)):
    print(f"{actions[value]:^5}: {count:>3}")


# Input for Neural Network
def make_input(obs, unit_id):
    width, height = obs["width"], obs["height"]
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}

    b = np.zeros((20, 32, 32), dtype=np.float32)

    for update in obs["updates"]:
        strs = update.split(" ")
        input_identifier = strs[0]

        if input_identifier == "u":
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            if unit_id == strs[3]:
                # Position and Cargo
                b[:2, x, y] = (1, (wood + coal + uranium) / 100)
            else:
                # Units
                team = int(strs[2])
                cooldown = float(strs[6])
                idx = 2 + (team - obs["player"]) % 2 * 3
                b[idx : idx + 3, x, y] = (
                    1,
                    cooldown / 6,
                    (wood + coal + uranium) / 100,
                )
        elif input_identifier == "ct":
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            idx = 8 + (team - obs["player"]) % 2 * 2
            b[idx : idx + 2, x, y] = (1, cities[city_id])
        elif input_identifier == "r":
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{"wood": 12, "coal": 13, "uranium": 14}[r_type], x, y] = amt / 800
        elif input_identifier == "rp":
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            b[15 + (team - obs["player"]) % 2, :] = min(rp, 200) / 200
        elif input_identifier == "c":
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10

    # Day/Night Cycle
    b[17, :] = obs["step"] % 40 / 40
    # Turns
    b[18, :] = obs["step"] / 360
    # Map Size
    b[19, x_shift : 32 - x_shift, y_shift : 32 - y_shift] = 1

    return b


class LuxDataset(Dataset):
    def __init__(self, obses, samples):
        self.obses = obses
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_id, unit_id, action = self.samples[idx]
        obs = self.obses[obs_id]
        state = make_input(obs, unit_id)

        return state, action


def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs):
    global best_acc
    global global_epoch
    global_epoch += 1

    for epoch in range(num_epochs):
        model.cuda()

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_acc = 0

            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                states = item[0].cuda().float()
                actions = item[1].cuda().long()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    policy = model(states)
                    loss = criterion(policy, actions)
                    _, preds = torch.max(policy, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(policy)
                    epoch_acc += torch.sum(preds == actions.data)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / data_size

            print(
                f"Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}"
            )

        if epoch_acc > best_acc:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, 20, 32, 32))
            traced.save(f"{MODEL_DIR}/model_all_luxnet_{epoch_acc}.pth")
            best_acc = epoch_acc


from luxnet_model import LuxNet

model = LuxNet()

model = torch.jit.load(FINE_TUNE_MODEL)

train, val = train_test_split(samples, test_size=0.1, random_state=12, stratify=labels)
batch_size = 64
train_loader = DataLoader(
    LuxDataset(obses, train), batch_size=batch_size, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    LuxDataset(obses, val), batch_size=batch_size, shuffle=False, num_workers=2
)
dataloaders_dict = {"train": train_loader, "val": val_loader}
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)


# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
# train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=12)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
# train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=10)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=7)


scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
best_acc = 0.0
global_epoch = 0
for n in range(40):
    print("Learning with lr :", optimizer.state_dict()["param_groups"][0]["lr"])
    train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=1)
    # We set the LR decayed every epoch
    scheduler.step()
