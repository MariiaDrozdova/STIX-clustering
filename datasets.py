from astropy.io import fits
import glob
import os
import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from transformers import AutoFeatureExtractor
from general import DATA_PATH, TEST_PATH

    
def load_filenames(filepath="saved_filtered_filenames_final.txt"):
    with open(filepath, 'r') as file:
        filenames = [line.strip() for line in file.readlines()]
    new_names = []
    for file_name in filenames:
        new_file_name = file_name.replace("stix_data/", f"{DATA_PATH}/stix_data/")
        new_names.append(new_file_name)
    return new_names

def load_filenames_inverse(filepath="saved_filtered_filenames_final.txt"):
    with open(filepath, 'r') as file:
        filenames = [line.strip() for line in file.readlines()]
    new_names = []
    all_filenames = os.listdir(f"{DATA_PATH}/stix_data/")
    for file_name in all_filenames:
        cur_name = "../../scratch/data/stix_data/" + file_name
        if cur_name not in filenames:
            new_file_name = cur_name.replace("data/stix_data/", f"{DATA_PATH}/stix_data/")
            new_names.append(new_file_name)
    return new_names

def find_all_fits_files(fits_folder):
    fits_files_list = load_filenames()
    return fits_files_list

def normalize_standard(data):
    data[data < -1 + 1e-5] = -1 + 1e-5
    data = np.log(1 + data)
    data_max = np.max(data, axis=(0,1), keepdims=True)
    data_min = np.min(data, axis=(0,1), keepdims=True)
    data = (data - data_min) / (data_max - data_min + 1e-15)
    return data

def preprocess_clip_wrapper(data, preprocess_func):
    data = normalize_standard(data)
    data = data*255
    pil_image = Image.fromarray(np.uint8(data)) 
    image = preprocess_func(pil_image).unsqueeze(0)
    return image

def preprocess_transformers_wrapper(data, feature_extractor):
    data = normalize_standard(data)
    data = data*255
    data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
    pil_image = Image.fromarray(np.uint8(data), "RGB") 
    image = feature_extractor(pil_image, return_tensors="pt")['pixel_values']
    return image
    
def preprocess_dino(data):
    data = normalize_standard(data)
    image = torch.tensor(data, dtype=torch.float32)
    image = image.repeat(1, 3, 1, 1) 
    IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1)
    IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1)
    image = (image - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD
    return image

class IJEPA():
    def __init__(self,path="ijepa_stix.npy"):
        self.features = torch.tensor(np.load(path), dtype=torch.float32)
        print(path, self.features.shape)
        
    def to(self, device):
        self.features = self.features.to(device)
        return self

    def forward(self,**kwargs):
        return self.features

        
class SolarDataset(Dataset):
    def __init__(self, fits_files_list, preprocess_func=lambda x : x, other_data_source=None, interpolate_size=None, classes=None, model=None):
        self.fits_files_list = fits_files_list
        self.preprocess_func = preprocess_func
        self.interpolate_size = interpolate_size
        self.classes = classes
        # Set device priority: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.device=device
        self.model = model
        if self.model is not None:
            self.model = model.to(self.device)
        self.features = None
        
    def __len__(self):
        return len(self.fits_files_list)
    
    def __getitem__(self, idx):
        fits_file = self.fits_files_list[idx]
        
        try:
            photon_count = int(fits_file.split('_')[-3]) 
        except:
            photon_count = -1
        
        # Extracting the thermal component (energy range) from the filename
        filename_parts = fits_file.split('_')
        for part in filename_parts:
            if 'keV' in part:
                thermal_component = part
                break
        else:
            thermal_component = 'Unknown'  # Fallback in case the pattern is not found

        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            cut_n=40
            data = hdul[0].data[cut_n:257-cut_n,cut_n:257-cut_n]
            date_obs = header.get('DATE_OBS', -1)
            rsun_obs = header.get('RSUN_OBS', -1)
            orbit_info = {"dsun_obs": header.get('DSUN_OBS', -1)}
            
        # Ensure the data is in the native byte order
        if data.dtype.byteorder not in ('=', '|'):  
            data = data.byteswap().newbyteorder()

        data = self.preprocess_func(data)
        if not isinstance(data, torch.Tensor):
            if len(data.shape) == 2:
                data = data[np.newaxis, ...]
            data = torch.tensor(data, dtype=torch.float32)
            data = data.unsqueeze(0)
            
        if self.interpolate_size is not None:
            data = torch.nn.functional.interpolate(
                data, 
                size=self.interpolate_size, 
                mode='bilinear', 
                align_corners=False
            )

        data = data.squeeze(0)
        
        sample = {
            "time": date_obs,
            "photon_counts": photon_count,
            "orbit_info": orbit_info,
            "rsun_obs": rsun_obs,
            "thermal_component": thermal_component,
            "data": data,
            "class": -1,
            "filename": fits_file
        }
        
        if self.classes is not None:
            sample["class"] = int(self.classes[idx])
            
        if self.features is not None:
            features = self.features[idx]
            features = features.reshape(1,-1)
            sample["features"] = features

        return sample 
    
    def assign_features(self, dataloader):
        import tqdm
        features_list = []
        if isinstance(self.model, IJEPA):
            self.features = self.model.forward()
            return    
        for i, batch in tqdm.tqdm(enumerate(dataloader)):
            data = batch["data"]
            if self.model is not None:
                if hasattr(self.model, 'encode_image'):
                    features =self.model.encode_image(data.to(self.device))
                else:
                    features =self.model(data.to(self.device))
            else:
                features = data
            if hasattr(features, 'last_hidden_state'):
                features = features.last_hidden_state
            features_list.append(features.cpu().detach().numpy())
        self.features = np.vstack(features_list)

class SolarDatasetIJEPA(Dataset):
    def __init__(self, fits_files_list, preprocess_func=lambda x : x, other_data_source=None, interpolate_size=None, classes=None, model=None, images=None, main_line="ijepa_imgs_ijepa",extra_line=""):
        self.images = np.load(f"{main_line}{extra_line}.npy").reshape(-1, 3, 224, 224)
        print("SolarDatasetIJEPA:", self.images.shape)
        self.fits_files_list = [str(i)+".fits" for i in range(len(self.images))]
        self.preprocess_func = preprocess_func
        self.interpolate_size = interpolate_size
        self.classes = classes
        # Set device priority: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.device=device
        self.model = model
        print("in init :", self.model)
        if self.model is not None:
            self.model = model.to(self.device)
        self.features = None
        
    def __len__(self):
        return len(self.fits_files_list)
    
    def __getitem__(self, idx):
        fits_file = self.fits_files_list[idx]

        try:
            photon_count = int(fits_file.split('_')[-3])
        except:
            photon_count = -1
        
        filename_parts = fits_file.split('_')
        for part in filename_parts:
            if 'keV' in part:
                thermal_component = part
                break
        else:
            thermal_component = 'Unknown'

        data = self.images[idx:idx+1,]
        date_obs = -1
        rsun_obs = -1
        orbit_info = {"dsun_obs": -1}  
            
        if data.dtype.byteorder not in ('=', '|'):  
            data = data.byteswap().newbyteorder()

        data = self.preprocess_func(data)
        if not isinstance(data, torch.Tensor):
            if len(data.shape) == 2:
                data = data[np.newaxis, ...]
            data = torch.tensor(data, dtype=torch.float32)
            data = data.unsqueeze(0)
            
        if self.interpolate_size is not None:
            data = torch.nn.functional.interpolate(
                data, 
                size=self.interpolate_size, 
                mode='bilinear', 
                align_corners=False
            )

        data = data.squeeze(0)
        
        sample = {
            "time": date_obs,
            "photon_counts": photon_count,
            "orbit_info": orbit_info,
            "rsun_obs": rsun_obs,
            "thermal_component": thermal_component,
            "data": data,
            "class": -1,
            "filename": fits_file
        }
        
        if self.classes is not None:
            sample["class"] = int(self.classes[idx])
            
        if self.features is not None:
            features = self.features[idx]
            features = features.reshape(1,-1)
            sample["features"] = features
        return sample 
    
    def assign_features(self, dataloader):
        import tqdm
        features_list = []
        if isinstance(self.model, IJEPA):
            self.features = self.model.forward()
            return    
        for i, batch in tqdm.tqdm(enumerate(dataloader)):
            data = batch["data"]
            if self.model is not None:
                if hasattr(self.model, 'encode_image'):
                    features =self.model.encode_image(data.to(self.device))
                else:
                    features =self.model(data.to(self.device))
            else:
                features = data
            features_list.append(features.cpu().detach().numpy())
        self.features = np.vstack(features_list)
        
def load_file_names_and_classes_for_test(parent_folder= f"{TEST_PATH}/stx_reconstructions"):
    test_fits_files = []
    classes = []
    def display_fits(fits_files, directory):
        for file in fits_files:
            if "clean" not in file:
                continue
            if "no_resid" in file:
                continue
            filepath = os.path.join(directory, file)
            test_fits_files.append(filepath)
            classes.append(directory.split("/")[-2])
    
    for folder in os.listdir(parent_folder):
        directory = f"{parent_folder}/{folder}/"
        try:
            fits_files = os.listdir(directory)
            display_fits(fits_files, directory)
        except NotADirectoryError:
            pass
            #print(f"skipping {directory} as it is not a directory")
    return test_fits_files, classes 

def prepare_dataloaders(mode, im_size=256, batch_size=4, inverse=False):
    if inverse:
        train_fits_files=load_filenames_inverse()
    else:
        train_fits_files=find_all_fits_files("../../scratch/data/stix_data/")
    test_fits_files, classes = load_file_names_and_classes_for_test()
    
    if mode=="clip":
        import open_clip
        from functools import partial
        
        im_size=224
        
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32')
            
        train_dataset = SolarDataset(
            fits_files_list=train_fits_files,
            interpolate_size=im_size,
            preprocess_func=partial(preprocess_clip_wrapper, preprocess_func=preprocess),
            model=model,
        )
        test_dataset = SolarDataset(
            fits_files_list=test_fits_files,
            interpolate_size=im_size,
            preprocess_func=partial(preprocess_clip_wrapper, preprocess_func=preprocess),
            model=model,
            classes=classes,
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    elif mode=='ijepa':
        im_size=224
        model = IJEPA()

        train_dataset = SolarDatasetIJEPA(
            fits_files_list=train_fits_files,
            interpolate_size=im_size,
            preprocess_func=lambda x : torch.tensor(x, dtype=torch.float32),
            model=model,
        )
        model = IJEPA(path="ijepa_stix_test.npy")
        test_dataset = SolarDatasetIJEPA(
            fits_files_list=test_fits_files,
            interpolate_size=im_size,
            preprocess_func=lambda x : torch.tensor(x, dtype=torch.float32),
            model=model,
            classes=classes,
            extra_line="_test",
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
        
    elif mode=='mae_precomputed':
        im_size=224
        model = IJEPA(path="data/mae_train_features.npy")

        train_dataset = SolarDatasetIJEPA(
            fits_files_list=train_fits_files,
            interpolate_size=im_size,
            preprocess_func=lambda x : torch.tensor(x, dtype=torch.float32),
            main_line="data/mae_train_images",
            model=model,
        )
        model = IJEPA(path="data/mae_test_features.npy")
        test_dataset = SolarDatasetIJEPA(
            fits_files_list=test_fits_files,
            interpolate_size=im_size,
            preprocess_func=lambda x : torch.tensor(x, dtype=torch.float32),
            model=model,
            classes=classes,
            main_line="data/mae_test_images",
            extra_line="",
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    
    elif mode=="msn-large":
        from functools import partial
        from transformers import AutoFeatureExtractor, ViTMSNModel
        
        im_size=224
        
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/vit-msn-large")
        model = ViTMSNModel.from_pretrained("facebook/vit-msn-large")
  
        train_dataset = SolarDataset(
            fits_files_list=train_fits_files,
            interpolate_size=im_size,
            preprocess_func=partial(preprocess_transformers_wrapper, feature_extractor=feature_extractor),
            model=model,
        )
        test_dataset = SolarDataset(
            fits_files_list=test_fits_files,
            interpolate_size=im_size,
            preprocess_func=partial(preprocess_transformers_wrapper, feature_extractor=feature_extractor),
            model=model,
            classes=classes,
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    elif mode=="mae":
        from functools import partial
        from transformers import AutoFeatureExtractor, ViTMSNModel
        
        im_size=224
        
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/vit-mae-large")
        model = ViTMSNModel.from_pretrained("facebook/vit-mae-large")
  
        train_dataset = SolarDataset(
            fits_files_list=train_fits_files,
            interpolate_size=im_size,
            preprocess_func=partial(preprocess_transformers_wrapper, feature_extractor=feature_extractor),
            model=model,
        )
        test_dataset = SolarDataset(
            fits_files_list=test_fits_files,
            interpolate_size=im_size,
            preprocess_func=partial(preprocess_transformers_wrapper, feature_extractor=feature_extractor),
            model=model,
            classes=classes,
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
        
    elif mode=="dino":
        im_size=224
        
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            
        train_dataset = SolarDataset(
            fits_files_list=train_fits_files,
            interpolate_size=im_size,
            preprocess_func=preprocess_dino,
            model=model,
        )
        test_dataset = SolarDataset(
            fits_files_list=test_fits_files,
            interpolate_size=im_size,
            preprocess_func=preprocess_dino,
            model=model,
            classes=classes,
        )

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
    
    elif mode=="standard":
        
        train_dataset = SolarDataset(
            fits_files_list=train_fits_files,
            interpolate_size=im_size,
            preprocess_func=normalize_standard,
        )
        test_dataset = SolarDataset(
            fits_files_list=test_fits_files,
            interpolate_size=im_size,
            preprocess_func=normalize_standard,
            classes=classes,
        )

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    else:
        print(f"No such an option: {mode}")
        
    train_dataset.assign_features(train_dataloader)
    test_dataset.assign_features(test_dataloader)
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader
