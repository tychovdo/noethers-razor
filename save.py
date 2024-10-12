import pickle

from pathlib import Path

def save_model(data, args, name, save_path='checkpoints'):
    path = Path(save_path)

    print(f"Saving data: {path / name}")
    path.mkdir(parents=True, exist_ok=True)

    with open(path /  f"{name}.data", 'wb') as data_file:
        pickle.dump(data, data_file)

    with open(path /  f"{name}.args", 'wb') as args_file:
        pickle.dump(args, args_file)
    
    print("Saved.")
