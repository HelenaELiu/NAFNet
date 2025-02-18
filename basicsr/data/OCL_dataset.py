import os
import random
from re import I, U
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from basicsr.data.data_util import *

dataset = "LF2016"
#dataset = "MVLF2018"
#dataset = "Sony"

if dataset == "LF2016": #9 categories
    categories = [
        "bikes", "buildings", "cars", "flowers_plants", "fruits_vegetables", 
        "general", "occlusions", "people", "reflective"
    ]
elif dataset == "MVLF2018": #30 categories
    categories = [
        "bamboo", "batteries", "benches", "bikes", "books",
        "bottles", "boxes", "buildings", "cables", "cacti", 
        "chairs", "coins", "cups", "drawers", "fire_hydrants", 
        "flowers", "glasses", "glue", "keyboards", "leaves", 
        "misc", "pens_and_pencils", "phones", "screws", "shelf", 
        "signs", "succulents", "tables", "tools", "trees"
    ]
elif dataset == "Sony": #5 categories
    categories = [
        "bear_back", "bear_front", "outdoor_night", "outdoor_selfie", "tree_flare"
    ]

ext = ".npy"
center_shift = f"0.0_center{ext}"

if dataset == "LF2016" or dataset == "MVLF2018":
    full_shift = f"0.0_full{ext}"
    ocl_views = [f"045{ext}", f"046{ext}", f"055{ext}", f"056{ext}"]
elif dataset == "Sony":
    ocl_views = [f"001{ext}", f"002{ext}", f"003{ext}", f"004{ext}"]

with_gt = True


class DataLoaderCenterViewsAndShiftToShift(Dataset):
    """
    Input: 2x2 shift + 4 views
    GT: 10x10 shift
    """

    def __init__(self, opt, test: bool = False):
        self.opt = opt
        self.paths = [] #(ocl shift, ocl views, gt)
        path = opt["dataroot_path"]
        num_scenes = 0

        with open(opt["test_list"], 'r') as f:
            test_files = json.load(f)

        for category in sorted(os.listdir(path)):
            if category not in categories:
                continue
            
            category_stack_path = f"{path}/{category}/focus_stack"
            category_block_path = f"{path}/{category}/lf_blocks"

            for scene in sorted(os.listdir(category_stack_path)):
                num_scenes += 1
                scene_stack_path = f"{category_stack_path}/{scene}"
                scene_block_path = f"{category_block_path}/{scene}"
                
                if test and scene_stack_path not in test_files:
                    continue
                if not test and scene_stack_path in test_files:
                    continue
                
                ocl_shift_path = f"{scene_stack_path}/{center_shift}"
                ocl_views_path = [f"{scene_block_path}/{view}" for view in ocl_views]
                if with_gt:
                    gt_path = f"{scene_stack_path}/{full_shift}"

                if not os.path.exists(ocl_shift_path):
                    print(f"{ocl_shift_path} does not exist")
                    continue
                if with_gt and not os.path.exists(gt_path):
                    print(f"{gt_path} does not exist")
                    continue
                
                if with_gt:
                    self.paths.append((ocl_shift_path, ocl_views_path, gt_path))
                else:
                    self.paths.append((ocl_shift_path, ocl_views_path))

        print(f"TOTAL NUMBER OF IMAGES: {num_scenes}") 
        print(f"NUMBER OF IMAGES IN SET: {len(self.paths)}") 
        
        if opt["random"]:
            random.shuffle(self.paths)
        
        if opt["size"] < len(self.paths):
            self.paths = self.paths[:opt["size"]]

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def preprocess_images( 
        img_path: str, img_size: tuple, crop: bool, ltm: bool, gamma: float):
        """
        Image Preprocessor
        Opens and formats the training image, used for both input and gt pairs
        """

        #img is RGB [0, 1] float32
        img = np.load(img_path)

        if ltm:
            mu = 1000
            img = np.log(1 + mu * img) / np.log(1 + mu)
        if gamma:
            img = img ** (1 / gamma)
            img = np.clip(img, 0, 1) 
        
        # Only accept downsampling or cropping
        if img_size and (img_size[0] < img.shape[0] or\
            img_size[1] < img.shape[1]):
            if crop:
                img = center_crop(img, img_size[1], img_size[0], "last")
            #else:
                #INTER_AREA because we are downsizing
                #img = cv2.resize(img, (img_size[0], img_size[1]), interpolation=cv2.INTER_AREA)
        
        # numpy (H, W, C) --> torch (1, C, H, W)
        torch_img = torch.from_numpy(img).float()
        torch_img = torch_img.permute(2, 0, 1) 

        return torch_img

    def __getitem__(self, idx: int):
        if with_gt:
            ocl_shift_path, ocl_views_path, gt_path = self.paths[idx]
        else:
            ocl_shift_path, ocl_views_path = self.paths[idx]
        
        img_size = (self.opt["img_height"], self.opt["img_width"])
        
        ocl_shift_image = self.preprocess_images(
            ocl_shift_path, img_size,  
            self.opt["crop"], self.opt["ltm"], self.opt["gamma"])
        
        ocl_views_images = [self.preprocess_images(
            ocl_view_path, img_size,
            self.opt["crop"], self.opt["ltm"], self.opt["gamma"]) 
            for ocl_view_path in ocl_views_path] 
        
        if with_gt:
            gt_image = self.preprocess_images(
                gt_path, img_size, 
                self.opt["crop"], self.opt["ltm"], self.opt["gamma"])
        
        #Stack the OCL shift and views
        ocl_stack = ocl_shift_image
        for ocl_view in ocl_views_images:
            ocl_stack = torch.cat((ocl_stack, ocl_view), dim = 0)
        
        if with_gt:
            return {'lq': ocl_stack, 'gt': gt_image} 
        else:
            return {'lq': ocl_stack} 





class DataLoaderCenterShiftToDefocusMap(Dataset):
    """
    Input: 2x2 shift
    GT: 10x10 defocus map
    """

    def __init__(self, opt, test: bool = False):
        self.opt = opt
        self.paths = [] #(ocl shift, gt)
        path = opt["dataroot_path"]
        num_scenes = 0

        with open(opt["test_list"], 'r') as f:
            test_files = json.load(f)

        for category in sorted(os.listdir(path)):
            if category not in categories:
                continue

            category_stack_path = f"{path}/{category}/focus_stack"

            for scene in sorted(os.listdir(category_stack_path)):
                num_scenes += 1
                scene_stack_path = f"{category_stack_path}/{scene}"

                if test and scene_stack_path not in test_files:
                    continue
                if not test and scene_stack_path in test_files:
                    continue

                ocl_shift_path = f"{scene_stack_path}/{center_shift}"
                if with_gt:
                    gt_path = f"{path}/defocus_maps/{scene}.png"
                
                if not os.path.exists(ocl_shift_path): 
                    print(f"{ocl_shift_path} does not exist")
                    continue
                if with_gt and not os.path.exists(gt_path): 
                    print(f"{gt_path} does not exist")
                    continue

                if with_gt:
                    self.paths.append((ocl_shift_path, gt_path))
                else:
                    self.paths.append((ocl_shift_path))
        
        print(f"TOTAL NUMBER OF IMAGES: {num_scenes}")
        print(f"NUMBER OF IMAGES IN SET: {len(self.paths)}")

        if opt["random"]:
            random.shuffle(self.paths)
        
        if opt["size"] < len(self.paths):
            self.paths = self.paths[:opt["size"]]

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def preprocess_images( 
        img_path: str, img_size: tuple, crop: bool, ltm: bool, gamma: float):
        """
        Image Preprocessor
        Opens and formats the training image, used for both input and gt pairs
        """
        
        if img_path.endswith(".npy"): #ocl data
            img = np.load(img_path)
        elif img_path.endswith(".png"): #defocus map
            img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1) #changes from (h, w) to (h, w, 1)
            img = np.float32(img) / 255

        if ltm:
            mu = 1000
            img = np.log(1 + mu * img) / np.log(1 + mu)
        if gamma:
            img = img ** (1/float(gamma))
            img = np.clip(img, 0, 1) 
            
        # numpy (H, W, C) --> torch (1, C, H, W)
        torch_img = torch.from_numpy(img).float()
        torch_img = torch_img.permute(2, 0, 1) 

        return torch_img

    def __getitem__(self, idx: int):
        if with_gt:
            ocl_shift_path, gt_path = self.paths[idx]
        else:
            ocl_shift_path = self.paths[idx]
        
        img_size = (self.opt["img_height"], self.opt["img_width"])

        ocl_shift_image = self.preprocess_images(
            ocl_shift_path, img_size, 
            self.opt["crop"], self.opt["ltm"], self.opt["gamma"])
        
        if with_gt:
            gt_image = self.preprocess_images(
                gt_path, img_size,
                self.opt["crop"], self.opt["ltm"], self.opt["gamma"])

        if with_gt:
            return {'lq': ocl_shift_image, 'gt': gt_image}
        else:
            return {'lq': ocl_shift_image}





class DataLoaderCenterViewsAndShiftToDefocusMap(Dataset):
    """
    Input: 2x2 shift + 4 views
    GT: 10x10 defocus map
    """

    def __init__(self, opt, test: bool = False):
        self.opt = opt
        self.paths = [] #(ocl shift, ocl views, gt)
        path = opt["dataroot_path"]
        num_scenes = 0

        with open(opt["test_list"], 'r') as f:
            test_files = json.load(f)

        for category in sorted(os.listdir(path)):
            if category not in categories:
                continue

            category_stack_path = f"{path}/{category}/focus_stack"
            category_block_path = f"{path}/{category}/lf_blocks"

            for scene in sorted(os.listdir(category_stack_path)):
                num_scenes += 1
                scene_stack_path = f"{category_stack_path}/{scene}"
                scene_block_path = f"{category_block_path}/{scene}"

                if test and scene_stack_path not in test_files:
                    continue
                if not test and scene_stack_path in test_files:
                    continue

                ocl_shift_path = f"{scene_stack_path}/{center_shift}"
                ocl_views_path = [f"{scene_block_path}/{view}" for view in ocl_views]
                if with_gt:
                    gt_path = f"{path}/defocus_maps/{scene}.png"
                
                if not os.path.exists(ocl_shift_path): 
                    print(f"{ocl_shift_path} does not exist")
                    continue
                if with_gt and not os.path.exists(gt_path): 
                    print(f"{gt_path} does not exist")
                    continue

                if with_gt:
                    self.paths.append((ocl_shift_path, ocl_views_path, gt_path))
                else:
                    self.paths.append((ocl_shift_path, ocl_views_path))
        
        print(f"TOTAL NUMBER OF IMAGES: {num_scenes}")
        print(f"NUMBER OF IMAGES IN SET: {len(self.paths)}")

        if opt["random"]:
            random.shuffle(self.paths)
        
        if opt["size"] < len(self.paths):
            self.paths = self.paths[:opt["size"]]

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def preprocess_images( 
        img_path: str, img_size: tuple, crop: bool, ltm: bool, gamma: float):
        """
        Image Preprocessor
        Opens and formats the training image, used for both input and gt pairs
        """
        
        if img_path.endswith(".npy"): #ocl data
            img = np.load(img_path)
        elif img_path.endswith(".png"): #defocus map
            img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1) #changes from (h, w) to (h, w, 1)
            img = np.float32(img) / 255

        if ltm:
            mu = 1000
            img = np.log(1 + mu * img) / np.log(1 + mu)
        if gamma:
            img = img ** (1 / float(gamma))
            img = np.clip(img, 0, 1) 
        
        if crop:
            img = img[:img_size[0], :img_size[1], :]
            
        # numpy (H, W, C) --> torch (1, C, H, W)
        torch_img = torch.from_numpy(img).float()
        torch_img = torch_img.permute(2, 0, 1) 

        return torch_img

    def __getitem__(self, idx: int):
        if with_gt:
            ocl_shift_path, ocl_views_path, gt_path = self.paths[idx]
        else:
            ocl_shift_path, ocl_views_path = self.paths[idx]
        
        img_size = (self.opt["img_height"], self.opt["img_width"])

        ocl_shift_image = self.preprocess_images(
            ocl_shift_path, img_size, 
            self.opt["crop"], self.opt["ltm"], self.opt["gamma"])
        
        ocl_views_images = [self.preprocess_images(
            ocl_view_path, img_size,
            self.opt["crop"], self.opt["ltm"], self.opt["gamma"])
            for ocl_view_path in ocl_views_path]
        
        if with_gt:
            gt_image = self.preprocess_images(
                gt_path, img_size,
                self.opt["crop"], self.opt["ltm"], self.opt["gamma"])
        
        #Stack the OCL shift and views
        ocl_stack = ocl_shift_image
        for ocl_view in ocl_views_images:
            ocl_stack = torch.cat((ocl_stack, ocl_view), dim = 0)

        if with_gt:
            return {'lq': ocl_stack, 'gt': gt_image}
        else:
            return {'lq': ocl_stack}
        
'''
class DataLoaderCenterViewsAndShiftAndDepthToShift(Dataset):
    """
    Input: 2x2 shift + 4 Views + Depth map
    GT: 10x10 shift
    """

    def __init__(self, opt, test: bool = False):
        self.opt = opt
        self.paths = [] #(ocl shift, ocl views, depth map, gt)
        path = opt["dataroot_path"]

        with open(opt["test_list"], 'r') as f:
            test_files = json.load(f)
        
        scene_num = 0
        for category in sorted(os.listdir(path)):
            if category not in categories:
                continue

            category_stack_path = f"{path}/{category}/focus_stack"
            category_block_path = f"{path}/{category}/lf_blocks"
            category_depth_path = f"{path}/{category}/depth_images"
            
            for scene in sorted(os.listdir(category_stack_path)):
                scene_num += 1
                scene_stack_path = f"{category_stack_path}/{scene}"
                scene_block_path = f"{category_block_path}/{scene}"
                
                if test and scene_stack_path not in test_files:
                    continue
                if not test and scene_stack_path in test_files:
                    continue
                
                ocl_scene_path = f"{scene_stack_path}/{center_view}"
                scene_depth_path = f"{category_depth_path}/{scene}".replace("_eslf", "_warp_depth.png")
                ocl_views_scene_path = [f"{scene_block_path}/{v}" for v in ocl_views]
                gt_scene_path = f"{scene_stack_path}/{full_view}"
                
                if not os.path.exists(ocl_scene_path) or \
                    not os.path.exists(gt_scene_path):
                    print(f"{ocl_scene_path} or {gt_scene_path} does not exist")
                    continue
                
                self.paths.append(
                    (ocl_scene_path, ocl_views_scene_path, scene_depth_path, gt_scene_path)
                )
        
        print(f"VISITED SCENES: {scene_num}") 
        print(f"NUMBER OF FILES: {len(self.paths)}") 
        
        if opt["random"]:
            random.shuffle(self.paths)
        
        if opt["size"] < len(self.paths):
            self.paths = self.paths[:opt["size"]]

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def preprocess_images( 
        img_path: str,
        img_size: tuple,
        crop: bool,
        ltm: bool,
        gamma: float,
        norm: bool = False,
        gray: bool = False):
        """Image Preprocessor

        Opens and formats the training image, used for both input and gt pairs
        """
        
        if img_path.endswith(".npy"): #RGB [0, 1] float32
            img = np.load(img_path)
        elif img_path.endswith(".png"): #depth map
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if norm: 
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
        if ltm:
            mu = 1000
            img = np.log(1 + mu * img) / np.log(1 + mu)
        
        if gamma:
            img = img ** (1 / gamma)
            img = np.clip(img, 0, 1) 
        
        # Only accept downsampling or cropping
        if img_size:
            if crop:
                img = center_crop(img, img_size[1], img_size[0], "last")
            else:
                img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)
        
        if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)

        # numpy (H,W,C) --> torch (1, C, H, W)
        torch_img = torch.from_numpy(img).float()
        torch_img = torch_img.permute(2, 0, 1) 

        return torch_img

    def __getitem__(self, idx: int):
        ocl_shift, ocl_views, depth_view, gt = self.paths[idx]
        img_size = (self.opt["img_height"], self.opt["img_width"])
        
        ocl_shift_image = self.preprocess_images(
            ocl_shift, img_size,  
            self.opt["crop"], self.opt["ltm"],
            self.opt["gamma"])
        
        ocl_views_images = [self.preprocess_images(
            ocl_view, img_size,
            self.opt["crop"], self.opt["ltm"],
            self.opt["gamma"]) 
            for ocl_view in ocl_views] 
        
        depth_image = self.preprocess_images(
            depth_view, img_size,
            False, self.opt["ltm"],
            self.opt["gamma"],
            norm=True)
        
        gt_image = self.preprocess_images(
            gt, img_size, 
            self.opt["crop"], self.opt["ltm"],
            self.opt["gamma"])
        
        #Stack OCL Image
        ocl_stack = ocl_shift_image
        for ocl_view in ocl_views_images:
            ocl_stack = torch.cat((ocl_stack, ocl_view), dim=0)
        ocl_stack = torch.cat((ocl_stack, depth_image), dim=0)

        return {'lq': ocl_stack, 'gt': gt_image} 


class DataLoaderCenterShiftToShift(Dataset):
    """LF 2016 OCL -> LF Dataset for training.
    """

    def __init__(self, opt):
        super(DataLoaderCenterShiftToShift, self).__init__()
        
        self.opt = opt
        self.paths = [] #(ocl, gt)
        path = opt["dataroot_path"]

        with open(opt["test_list"], 'r') as f:
            test_files = json.load(f)

        for category in sorted(os.listdir(path)):
            if category not in categories:
                continue
            category_stack_path = f"{path}/{category}/focus_stack"
            for scene in sorted(os.listdir(category_stack_path)):
                scene_path = f"{category_stack_path}/{scene}"
                if scene_path in test_files:
                    continue
                ocl_scene_path = f"{scene_path}/{center_view}"
                gt_scene_path = f"{scene_path}/{full_view}"
                if not os.path.exists(ocl_scene_path) or \
                    not os.path.exists(gt_scene_path):
                    continue
                self.paths.append(
                    (ocl_scene_path, gt_scene_path)
                )
                print(ocl_scene_path, gt_scene_path)
                
        if opt["random"]:
            random.shuffle(self.paths)
        
        if opt["size"] < len(self.paths):
            self.paths = self.paths[:opt["size"]]

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def preprocess_images( 
        img_path: str,
        img_size: tuple,
        crop: bool,
        ltm: bool,
        gamma: float):
        """Image Preprocessor

        Opens and formats the training image, used for both input and gt pairs
        """
        
        if img_path.endswith(".npy"): #RGB [0, 1] float32
            img = np.load(img_path)
        elif img_path.endswith(".png"): #BGR [0, 65535] Uint16 --> RGB [0, 1] float32
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = np.float32(img)[:,:,::-1] / 65535

        if ltm:
            mu = 1000
            img = np.log(1 + mu * img) / np.log(1 + mu)
        if gamma:
            img = img ** (1/float(gamma))
            img = np.clip(img, 0, 1) 
        

        #if img_size and img_size < img.shape[0]:
        #    if crop:
        #        img = center_crop(img, img_size, img_size, "last")
        #    else:
        #        #INTER_AREA because we are downsizing
        #        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        
        # numpy (H,W,C) --> torch (1, C, H, W)
        torch_img = torch.from_numpy(img).float()
        torch_img = torch_img.permute(2, 0, 1) 

        return torch_img

    def __getitem__(self, idx: int):

        ocl, gt = self.paths[idx]
        img_size = (self.opt["img_height"], self.opt["img_width"])
        ocl_image = self.preprocess_images(
            ocl, img_size, 
            self.opt["crop"], self.opt["ltm"],
            self.opt["gamma"])
        gt_image = self.preprocess_images(
            gt, img_size,
            self.opt["crop"], self.opt["ltm"],
            self.opt["gamma"])

        return ocl_image, gt_image


class DataLoaderCenterViewsToShift(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.paths = [] #(ocl, gt)
        path = opt["dataroot_path"]

        with open(opt["test_list"], 'r') as f:
            test_files = json.load(f)

        for category in sorted(os.listdir(path)):
            if category not in categories:
                continue
            category_stack_path = f"{path}/{category}/focus_stack"
            category_block_path = f"{path}/{category}/lf_blocks"
            for scene in sorted(os.listdir(category_stack_path)):
                scene_stack_path = f"{category_stack_path}/{scene}"
                scene_block_path = f"{category_block_path}/{scene}"
                if scene_stack_path in test_files:
                    continue
                ocl_scene_path = f"{scene_stack_path}/{center_view}"
                ocl_views_scene_path = [f"{scene_block_path}/{v}" for v in ocl_views]
                print(ocl_views_scene_path)
                gt_scene_path = f"{scene_stack_path}/{full_view}"
                if not os.path.exists(ocl_scene_path) or \
                    not os.path.exists(gt_scene_path):
                    continue
                self.paths.append(
                    (ocl_scene_path, ocl_views_scene_path, gt_scene_path)
                )
                print(ocl_scene_path, gt_scene_path)
                
        if opt["random"]:
            random.shuffle(self.paths)
        
        if opt["size"] < len(self.paths):
            self.paths = self.paths[:opt["size"]]

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def preprocess_images( 
        img_path: str,
        img_size: tuple,
        crop: bool,
        ltm: bool, 
        gamma: float):
        """Image Preprocessor

        Opens and formats the training image, used for both input and gt pairs
        """
        
        if img_path.endswith(".npy"): #RGB [0, 1] float32
            img = np.load(img_path)
        elif img_path.endswith(".png"): #BGR [0, 65535] Uint16 --> RGB [0, 1] float32
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = np.float32(img)[:,:,::-1] / 65535

        if ltm:
            mu = 1000
            img = np.log(1 + mu * img) / np.log(1 + mu)
        if gamma: 
            img = img ** (1 / gamma)
        # Only accept downsampling or cropping
        if img_size and (img_size[0] < img.shape[0] or\
            img_size[1] < img.shape[1]):
            if crop:
                img = center_crop(img, img_size[1], img_size[0], "last")
            else:
                #INTER_AREA because we are downsizing
                img = cv2.resize(img, (img_size[0], img_size[1]), interpolation=cv2.INTER_AREA)
        
        # numpy (H,W,C) --> torch (1, C, H, W)
        torch_img = torch.from_numpy(img).float()
        torch_img = torch_img.permute(2, 0, 1) 

        return torch_img

    def __getitem__(self, idx: int):

        _, ocl_views, gt = self.paths[idx]
        img_size = (self.opt["img_height"], self.opt["img_width"])
        ocl_views_images = [self.preprocess_images(
            ocl_view, img_size,
            self.opt["crop"], self.opt["ltm"],
            self.opt["gamma"]) 
            for ocl_view in ocl_views] 
        gt_image = self.preprocess_images(
            gt, img_size, 
            self.opt["crop"], self.opt["ltm"],
            self.opt["gamma"])
        
        #Stack OCL Image
        ocl_stack = torch.tensor(())
        for ocl_view in ocl_views_images:
            ocl_stack = torch.cat((ocl_stack, ocl_view), dim=0)

        return ocl_stack, gt_image


class DataLoaderCenterViewDiffAndShiftToShift(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.paths = [] #(ocl, gt)
        path = opt["dataroot_path"]

        with open(opt["test_list"], 'r') as f:
            test_files = json.load(f)

        for category in sorted(os.listdir(path)):
            if category not in categories:
                continue
            category_stack_path = f"{path}/{category}/focus_stack"
            category_block_path = f"{path}/{category}/lf_blocks"
            for scene in sorted(os.listdir(category_stack_path)):
                scene_stack_path = f"{category_stack_path}/{scene}"
                scene_block_path = f"{category_block_path}/{scene}"
                if scene_stack_path in test_files:
                    continue
                ocl_scene_path = f"{scene_stack_path}/{center_view}"
                ocl_views_scene_path = [f"{scene_block_path}/{v}" for v in ocl_views]
                print(ocl_views_scene_path)
                gt_scene_path = f"{scene_stack_path}/{full_view}"
                if not os.path.exists(ocl_scene_path) or \
                    not os.path.exists(gt_scene_path):
                    continue
                self.paths.append(
                    (ocl_scene_path, ocl_views_scene_path, gt_scene_path)
                )
                print(ocl_scene_path, gt_scene_path)
                
        if opt["random"]:
            random.shuffle(self.paths)
        
        if opt["size"] < len(self.paths):
            self.paths = self.paths[:opt["size"]]

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def preprocess_images( 
        img_path: str,
        img_size: tuple,
        crop: bool,
        ltm: bool, 
        gamma: float):
        """Image Preprocessor

        Opens and formats the training image, used for both input and gt pairs
        """
        
        if img_path.endswith(".npy"): #RGB [0, 1] float32
            img = np.load(img_path)
        elif img_path.endswith(".png"): #BGR [0, 65535] Uint16 --> RGB [0, 1] float32
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = np.float32(img)[:,:,::-1] / 65535

        if ltm:
            mu = 1000
            img = np.log(1 + mu * img) / np.log(1 + mu)
        if gamma:
            img = img ** (1 / gamma)
        # Only accept downsampling or cropping
        if img_size and (img_size[0] < img.shape[0] or\
            img_size[1] < img.shape[1]):
            if crop:
                img = center_crop(img, img_size[1], img_size[0], "last")
            else:
                #INTER_AREA because we are downsizing
                img = cv2.resize(img, (img_size[0], img_size[1]), interpolation=cv2.INTER_AREA)

        # numpy (H,W,C) --> torch (1, C, H, W)
        torch_img = torch.from_numpy(img).float()
        torch_img = torch_img.permute(2, 0, 1) 

        return torch_img
    
    @staticmethod
    def preprocess_images_viewdiff(
        img_path: list, 
        img_size: tuple,
        crop: bool, 
        ltm: bool,
        gamma: float
    ):
        """List Image PreProcessor for view difference

        Wrapper for preprocess_images for a list
        """
        img_list = []
        for img_p in img_path:
            if img_path.endswith(".npy"): #RGB [0, 1] float32
                img = np.load(img_path)
            elif img_path.endswith(".png"): #BGR [0, 65535] Uint16 --> RGB [0, 1] float32
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                img = np.float32(img)[:,:,::-1] / 65535
            
            if ltm:
                mu = 1000
                img = np.log(1 + mu * img) / np.log(1 + mu)
            if gamma:
                img = img ** (1 / gamma)
                img = np.clip(img, 0, 1)

            # Only accept downsampling or cropping
            if img_size and (img_size[0] < img.shape[0] or\
                img_size[1] < img.shape[1]):
                if crop:
                    img = center_crop(img, img_size[1], img_size[0], "last")
                else:
                    #INTER_AREA because we are downsizing
                    img = cv2.resize(img, (img_size[0], img_size[1]), interpolation=cv2.INTER_AREA)

            img_list.append(img)
        
        imgs = np.stack(img_list, axis=0)

        view_diff = view_sum_diff(imgs)
        view_diff = np.repeat(view_diff[:,:,np.newaxis], [3], axis=-1)

        # numpy (H,W,C) --> torch (1, C, H, W)
        torch_img = torch.from_numpy(view_diff).float()
        torch_img = torch_img.permute(2, 0, 1)

        return torch_img  


    def __getitem__(self, idx: int):

        ocl_shift, ocl_views, gt = self.paths[idx]
        img_size = (self.opt["img_height"], self.opt["img_width"])
        ocl_shift_image = self.preprocess_images(
            ocl_shift, img_size,  
            self.opt["crop"], self.opt["ltm"],
            self.opt["gamma"])
        ocl_views_diff = self.preprocess_images_viewdiff(
            [ocl for ocl in ocl_views], img_size,
            self.opt["crop"], self.opt["ltm"],
            self.opt["gamma"]) 
        gt_image = self.preprocess_images(
            gt, img_size, 
            self.opt["crop"], self.opt["ltm"],
            self.opt["gamma"])
        
        #Stack OCL Image
        ocl_stack = torch.cat([ocl_views_diff, ocl_shift_image], dim=0)
        print(ocl_stack.size())

        return ocl_stack, gt_image
'''