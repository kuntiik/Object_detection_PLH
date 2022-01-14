import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from pathlib import Path
import yaml


def get_dataloader_mean_std(dataloader : DataLoader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in dataloader:
        data = torch.stack(data)
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    return mean, std

def _loader_to_folder(loader, path_im, path_lab):
    idx = 0
    for batch in loader:
        x, y = batch
        x[0].save(str(path_im) + "/" + str(idx) + ".png")
        width, height = x[0].size
        with open(str(path_lab) + "/" + str(idx) + ".txt", 'w') as f:
            for label, box in zip(y[0]['labels'], y[0]['boxes']):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                center_x = x1 + w/2
                center_y = y1 + h/2
                w /= width
                h /= height
                center_x /= width
                center_y /= height
                #faster rcnn ecpects first class to be 1
                label -= 1
                f.write(f"{label} {center_x} {center_y} {w} {h}\n")
        idx += 1


            


def generate_yolo_folders(module : LightningDataModule, target_folder : str):
    target_folder = Path(target_folder)
    _, _, test_size = module.hparams.data_split
    root = target_folder / module.name
    root.mkdir(parents=True, exist_ok=True)
    train_images_path = root / "images" / "train"
    val_images_path = root / "images" / "val"
    train_labels_path = root / "labels" / "train"
    val_labels_path = root / "labels" / "val"
    test_images_path = root / "images" / "test"
    test_labels_path = root / "labels" / "test"
    train_images_path.mkdir(parents=True, exist_ok=True)
    train_labels_path.mkdir(parents=True, exist_ok=True)
    val_images_path.mkdir(parents=True, exist_ok=True)
    val_labels_path.mkdir(parents=True, exist_ok=True)



    train_loader = module.train_dataloader()
    _loader_to_folder(train_loader, train_images_path, train_labels_path )
    val_loader = module.val_dataloader()
    _loader_to_folder(val_loader, val_images_path, val_labels_path )


    data = dict(
        train = str(train_images_path),
        val = str(val_images_path),
        nc = module.num_classes,
        names = ['decay']
    )

    if test_size > 0:
        test_images_path.mkdir(parents=True, exist_ok=True)
        test_labels_path.mkdir(parents=True, exist_ok=True)
        test_loader = module.test_dataloader()
        _loader_to_folder(test_loader, test_images_path, test_labels_path )
        data['test'] = str(test_images_path)

    with open(str(root) + "/" + module.name + ".yaml", 'w') as f:
        yaml.dump(data,f)
    

    




