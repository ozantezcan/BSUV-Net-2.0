"""
Data augmentation tools for chnagedetection type input and outputs

Definitions of the variables used in this code are as follows

CD input (dict): key                -> value (value=None if that field is not used)
                 "empty_bg_seg"     -> Foreground probability map of the empty background candidate.
                                       Size: (HxWx1), Type: float
                 "empty_bg"         -> Empty background candidate in RGB.
                                       Size: (HxWx3), Type: float
                 "recent_bg_seg"    -> Foreground probability map of the recent background candidate
                                       Size: (HxWx1), Type: float
                 "recent_bg"        -> Recent background candidate in RGB.
                                       Size: (HxWx3), Type: float
                 "current_fr_seg"   -> Foreground probability map of the current frame.
                                       Size: (HxWx1), Type: float
                 "current_fr"       -> Current frame in RGB.
                                       Size: (HxWx3), Type: float

CD output (Size: (HxWx1), Type: float): Background Segmentation Label for resepctive CD input.
                                        Follows the CDNet2014 format
"""

import torchvision.transforms as tvtf
import torch
import numpy as np
import cv2
from utils import augmentations as aug
from utils.data_loader import CDNet2014Loader

only_rgb_inputs = ["empty_bg", "recent_bg",  "current_fr"]
only_seg_inputs = ["empty_bg_seg", "recent_bg_seg",  "current_fr_seg"]

class AdditiveRandomIllumation:
    """Applies additive random illumination change to all frames and also increases
    illumination difference between the empty background and the current frame.

    Args:
        std_global (float, float): Standard deviation for the random illumination change
                                   for all color channels and between color channels.
        std_illdiff (float, float) (default=(0, 0): Standard deviation for the random
                illumination difference between the empty background and current frame
                for all color channels and between color channels.

    """
    def __init__(self, std_global, std_illdiff=(0, 0), debug=False):
        self.std_global = std_global
        self.std_illdiff = std_illdiff
        self.debug = debug

    def __call__(self, cd_inp, cd_out):
        """
        Args:
            cd_inp (CD input): Input to be converted
            cd_out (CD output): Output to be converted
        Returns:
            CD input: Updated CD input.
            CD output: Updated CD output.
        """
        illumination = (np.random.randn() * self.std_global[0]) + \
                       (np.random.randn(3) * self.std_global[1])

        if self.debug:
            print("Applying random illumination difference")

        for inp_type in only_rgb_inputs:
            if cd_inp[inp_type] is not None:
                cd_inp[inp_type] += illumination

        if cd_inp["empty_bg"] is not None:
            cd_inp["empty_bg"] += (np.random.randn() * self.std_illdiff[0]) +\
                                  (np.random.randn(3) * self.std_illdiff[1])

        return cd_inp, cd_out

class AdditiveNoise:
    """Adds gaussian noise to CD input

    Args:
        std (float): Standard deviation of the noise
    """

    def __init__(self, std_noise, debug=False):
        self.std_noise = std_noise
        self.debug = debug

    def __call__(self, cd_inp, cd_out):
        """
        Args:
            cd_inp (CD input): Input to be converted
            cd_out (CD output): Output to be converted
        Returns:
            CD input: Updated CD input.
            CD output: Updated CD output.
        """

        if self.debug:
            print("Applying random noise")

        h, w, c = cd_out.shape
        for inp_type in only_rgb_inputs:
            if cd_inp[inp_type] is not None:
                cd_inp[inp_type] += np.random.randn(h, w, 3) * self.std_noise

        return cd_inp, cd_out

class Resize:
    """Resizes CD input and CD output

    Args:
        out_dim ((int, int)): Target width and height
        interploation (optional): One of the methods from opencv2 interpolation methods.
                                  Default is cv2.INTER_LINEAR
    """
    def __init__(self, out_dim, interpolation=cv2.INTER_LINEAR):
        self.out_dim = out_dim
        self.interpolation = interpolation

    def __call__(self, cd_inp, cd_out):
        """
        Args:
            cd_inp (CD input): Input to be converted
            cd_out (CD output): Output to be converted
        Returns:
            CD input: Resized CD input.
            CD output: Resized CD output.
        """
        for inp_type, im in cd_inp.items():
            if im is not None:
                cd_inp[inp_type] = cv2.resize(im, self.out_dim, interpolation=self.interpolation)
                del im
        cd_out = cv2.resize(cd_out, self.out_dim, interpolation=self.interpolation)
        return cd_inp, cd_out

class CenterCrop:
    """ Extracts the center crop from CD input and CD output

    Args:
        out_dim ((int, int)): Target width and height of the crop
    """
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def __call__(self, cd_inp, cd_out):
        """
        Args:
            cd_inp (CD input): Input to be cropped
            cd_out (CD output): Output to be cropped
        Returns:
            CD input: Cropped CD input.
            CD output: Cropped CD output.
        """
        h, w, c = cd_out.shape
        i = int((w-self.out_dim[0])/2)
        j = int((h-self.out_dim[1])/2)
        for inp_type, im in cd_inp.items():
            if im is not None:
                cd_inp[inp_type] = im[j:j+self.out_dim[1], i:i+self.out_dim[0], :]
                del im

        cd_out = cd_out[j:j+self.out_dim[1], i:i+self.out_dim[0], :]
        return cd_inp, cd_out
        
class RandomCrop:
    """ Extracts a random crop from CD input and CD output

    Args:
        out_dim ((int, int)): Target width and height of the crop
    """
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def __call__(self, cd_inp, cd_out):
        """
        Args:
            cd_inp (CD input): Input to be cropped
            cd_out (CD output): Output to be cropped
        Returns:
            CD input: Cropped CD input.
            CD output: Cropped CD output.
        """
        h, w, c = cd_out.shape
        i = np.random.randint(low=0, high=w - self.out_dim[0])
        j = np.random.randint(low=0, high=h - self.out_dim[1])
        for inp_type, im in cd_inp.items():
            if im is not None:
                cd_inp[inp_type] = im[j:j+self.out_dim[1], i:i+self.out_dim[0], :]
                del im

        cd_out = cd_out[j:j+self.out_dim[1], i:i+self.out_dim[0], :]
        return cd_inp, cd_out

class RandomJitteredCrop:
    """ Extracts a random crop from CD input and CD output The output will have a jitter effect

    Args:
        out_dim ((int, int)): Target width and height of the crop
        max_jitter (int): Max number of pixels allowed to shift between background and recent frames (default 10)
        jitter_prob (float): probability of applying random jitter (default 0.5)
    """
    def __init__(self, out_dim, max_jitter=5, jitter_prob=1.0, debug=False):
        self.out_dim = out_dim
        self.max_jitter = max_jitter
        self.jitter_prob = jitter_prob
        self.debug = debug

    def __call__(self, cd_inp, cd_out):
        h, w, c = cd_out.shape
        if self.debug:
            print("Applying Jitter")
        if np.random.uniform() <= self.jitter_prob:
            max_jitter_w = min(self.max_jitter, int((w-self.out_dim[0]) / 2) - 1)
            max_jitter_h = min(self.max_jitter, int((h - self.out_dim[1]) / 2) - 1)
            j = np.random.randint(low=max_jitter_w, high=w - (self.out_dim[0] + max_jitter_w))
            i = np.random.randint(low=max_jitter_h, high=h - (self.out_dim[1] + max_jitter_h))
            empty_bg_offset = [np.random.randint(-max_jitter_h, max_jitter_h), np.random.randint(-max_jitter_w, max_jitter_w)]
            recent_bg_offset = [np.random.randint(-max_jitter_h, max_jitter_h), np.random.randint(-max_jitter_w, max_jitter_w)]
        else:
            j = np.random.randint(low=0, high=w - self.out_dim[0])
            i = np.random.randint(low=0, high=h - self.out_dim[1])
            empty_bg_offset = [0, 0]
            recent_bg_offset = [0, 0]

        for inp_type, im in cd_inp.items():
            if inp_type.startswith("empty_bg"):
                i_, j_ = i + empty_bg_offset[0], j + empty_bg_offset[1]
            elif inp_type.startswith("recent_bg"):
                i_, j_ = i + recent_bg_offset[0], j + recent_bg_offset[1]
            else:
                i_, j_ = i, j
            if im is not None:
                cd_inp[inp_type] = im[i_:i_+self.out_dim[1], j_:j_+self.out_dim[0], :]
                del im

        cd_out = cd_out[i:i+self.out_dim[1], j:j+self.out_dim[0], :]
        return cd_inp, cd_out

class RandomZoomCrop:
    """ Changes the background to zoomed out version

    Args:
        max_zoom_ratio_empty (float): Max allowed zoom_out ratio for empty reference frame (default 0.5)
        max_zoom_ratio_recent (float): Max allowed zoom_out ratio for recent reference frame (default 0.9)
        zoom_prob (float): probability of applying Zoom (default 0.5)
        num_frames (int): Number of frames to be averaged for reference frames (default 50)
        debug (boolean): Debuging purposes
    """
    def __init__(self, out_dim, max_zoom_ratio_empty=0.04, max_zoom_ratio_recent = 0.02, zoom_prob=1.0, num_frames=10, debug=False):
        self.centerCrop = CenterCrop(out_dim)
        self.max_zoom_ratio_empty = max_zoom_ratio_empty
        self.max_zoom_ratio_recent = max_zoom_ratio_recent
        self.zoom_prob = zoom_prob
        self.num_frames = num_frames
        self.debug = debug

    def __call__(self, cd_inp, cd_out):
        cd_inp, cd_out = self.centerCrop(cd_inp, cd_out)
        h, w, c = cd_out.shape
        if np.random.uniform() <= self.zoom_prob:
            
            zoom_ratio_recent = np.random.uniform(low=0.0, high=self.max_zoom_ratio_recent)
            zoom_ratio_empty = np.random.uniform(low=zoom_ratio_recent, high=self.max_zoom_ratio_empty)   

            if np.random.uniform() < 0.5:
                w_c, h_c = np.floor(w / (1 + (zoom_ratio_empty*self.num_frames))), np.floor(h / (1 + (zoom_ratio_empty*self.num_frames)))
                for inp_type in ["current_fr", "current_fr_seg"]:
                    im = cd_inp[inp_type]
                    if im is not None:
                        cd_inp[inp_type] = _resize(_centerCrop(im, w_c, h_c), w, h)
                cd_out = _resize(_centerCrop(cd_out, w_c, h_c), w, h)
            else:
                zoom_ratio_recent, zoom_ratio_empty = -zoom_ratio_recent, -zoom_ratio_empty
                w_c, h_c = w, h

            if self.debug:
                print(f'Zoom ratio empty = {zoom_ratio_empty}, Zoom ratio recent = {zoom_ratio_recent}')
            
            for inp_type in ["empty_bg", "empty_bg_seg", "recent_bg", "recent_bg_seg"]:
                im = cd_inp[inp_type]
                if im is not None:
                    im_transformed = im.copy()
                    if inp_type in ["empty_bg", "empty_bg_seg"]:
                        zoom_ratio = zoom_ratio_empty
                    else:
                        zoom_ratio = zoom_ratio_recent

                    for n in range(1, self.num_frames):
                        w_n, h_n = w_c * (1 + (n*zoom_ratio)), h_c * (1 + (n*zoom_ratio))
                        im_transformed += _resize(_centerCrop(im, w_n, h_n), w, h)

                    cd_inp[inp_type] = im_transformed / self.num_frames
                    del im
            

        return cd_inp, cd_out

class RandomPanCrop:
    """ Extracts a random crop from CD input and CD output The output will have a pan effect

    Args:
        out_dim ((int, int)): Target width and height of the crop
        max_pixel_shift (float): Max number of pixels allowed to shift for pan.
        num_frames_recent (int): Number of frames to be averaged for augemnted recent reference
        num_frames_empty (int): Number of frames to be averaged for augemnted empty reference
    """
    def __init__(self, out_dim, max_pixel_shift=5, num_frames_recent=10, num_frames_empty=20, debug=False):
        self.out_dim = out_dim
        self.max_pixel_shift = max_pixel_shift
        self.num_frames_recent = num_frames_recent
        self.num_frames_empty = num_frames_empty
        self.debug = debug

    def __call__(self, cd_inp, cd_out):
        h, w, _ = cd_out.shape
        if self.debug:
            print("Applying Random Pan")

        max_pixel_shift = min(self.max_pixel_shift, int((w-self.out_dim[0]) / self.num_frames_empty) - 1)
        pixel_shift = np.random.uniform(low=0, high=max_pixel_shift)
        j = np.random.randint(low=0, high=w - (self.out_dim[0] + (pixel_shift * self.num_frames_empty)))
        i = np.random.randint(low=0, high=h - self.out_dim[1])
        left_pan = np.random.randint(2) # left pan if 1, right pan if 0

        if self.debug:
            print("Left  Pan") if left_pan else print("Right Pan")

        for inp_type, im in cd_inp.items():

            if im is not None:
                if inp_type.startswith("empty_bg"):
                    panned_indices = [int(j + pixel_shift*k) for k in range(self.num_frames_empty)]
                elif inp_type.startswith("recent_bg"):
                    offset = 0 if left_pan else int(pixel_shift*(self.num_frames_empty - self.num_frames_recent))
                    panned_indices = [int(j + offset + pixel_shift*k) for k in range(self.num_frames_recent)]
                else:
                    panned_indices = [j if left_pan else int(j + pixel_shift*(self.num_frames_empty-1))]

                im_panned = np.zeros_like(im[:self.out_dim[1], :self.out_dim[0], :])
                for j_ in panned_indices:
                    im_panned += im[i:i+self.out_dim[1], j_:j_+self.out_dim[0]]
                cd_inp[inp_type] = im_panned / len(panned_indices)
                del im, im_panned

        offset = 0 if left_pan else int(pixel_shift*(self.num_frames_empty-1))
        cd_out = cd_out[i:i+self.out_dim[1], j+offset:j+offset+self.out_dim[0], :]
        return cd_inp, cd_out

class RandomMask:
    """ Extracts a random crop from CD input and CD output with a random mask

    Args:
        out_dim ((int, int)): Target width and height of the crop
        mask_dataset (dictionary {string:string}): Dcitionary of mask category and scenes
        empty_bg, recent_bg, seg_ch,selected_frames: Check docstring of CDNet2014Loader
    """
    def __init__(self, out_dim, dataloader_mask, mask_prob=1.0, debug=False):
        self.dataloader_mask = dataloader_mask
        self.mask_prob = mask_prob
        self.debug = debug

    def __call__(self, cd_inp, cd_out):
        if np.random.uniform() <= self.mask_prob:
            if self.debug:
                print("Applying Random Masking")
            sent_to_bg = np.random.randint(2)
            mask_inp, mask_label = next(iter(self.dataloader_mask))
            for inp_type, im in cd_inp.items():
                if (not inp_type.startswith('empty_bg')) and (im is not None):
                    masked_im = im.copy()
                    if sent_to_bg:
                        mask = mask_label[:, :, 0]
                        mask[cd_out[:, :, 0] == 1] = 0
                    else:
                        mask = mask_label[:, :, 0]
                    for k in range(masked_im.shape[-1]):
                        masked_im[:, :, k][mask == 1] = mask_inp[inp_type][:, :, k][mask == 1]
                    cd_inp[inp_type] = masked_im
                    del im
            cd_out[mask_label == 1] = 1
            
        return cd_inp, cd_out

      
class ToTensor:
    """ Converts CD input and CD output into tensors.
    Each defined element of CD input will be converted tensors and than concataneted in the
    following order of their definitions according the DocString. Size of the output tensor
    will be CxWxH where W, H are the spatial dimensions and C is the total number of channels
    in CD input (e.g if only empty_bg_seg, empty_bg, current_fr_seg, current_fr are defined
    (not None), ouput size will be (1+3+1+3)xWxH = 8xWxH)
    """
    def __call__(self, cd_inp, cd_out):
        """
        Args:
            cd_inp (CD input): Input to be converted
            cd_out (CD output): Output to be converted

        Returns:
            Tensor: Converted and concataneted CD input.
            Tensor: Converted CD output
        """
        inp_tensors = []
        for inp_type, im in cd_inp.items():
            if im is not None:
                inp_tensors.append(tvtf.ToTensor()(im.copy()))

        inp_tensor = torch.cat(inp_tensors, dim=0)

        return inp_tensor, tvtf.ToTensor()(cd_out)

class NormalizeTensor:
    """
    Normalizes input tensor channelwise using mean and std

    Args:
        mean_rgb ([_, _, _]): Sequence of means for RGB channels respectively
        std_rgb ([_, _, _]): Sequence of standard deviations for for RGB channels respectively
        mean_seg ([_]): Mean for segmentation channel
        std_seg ([_]): Standard deviation for segmentation channel
        segmentation_ch(bool): Bool for the usage of segmentation channel
    """
    def __init__(self, mean_rgb, mean_seg, std_rgb, std_seg, segmentation_ch=False):
        self.mean_rgb = mean_rgb
        self.std_rgb = std_rgb
        self.mean_seg = mean_seg
        self.std_seg = std_seg
        self.segmentation_ch = segmentation_ch

    def __call__(self, inp, out):
        """
        Args:
            inp (Tensor): Input tensor
            out (Tensor): Output Tensor

        Returns:
            Tensor: Normalized input tensor
            Tensor: Unchanged output tensor (only for cocistency in the code)
        """

        mean_period = self.mean_rgb.copy()
        std_period = self.std_rgb.copy()
        if self.segmentation_ch:
            mean_period = np.concatenate((self.mean_seg, mean_period))
            std_period = np.concatenate((self.std_seg, std_period))

        c, h, w = inp.size()
        num_frames = int(c / (3+(1*self.segmentation_ch)))

        mean_vec = np.concatenate([mean_period for _ in range(num_frames)])
        std_vec = np.concatenate([std_period for _ in range(num_frames)])
        inp_n = tvtf.Normalize(mean_vec, std_vec)(inp)
        return inp_n, out

def _centerCrop(im, w_, h_):
    """
    Take center_crop
    """
    h, w, _ = im.shape
    h_, w_ = int(h_), int(w_)
    i = int((w-w_)/2)
    j = int((h-h_)/2)
    return im[j:j+h_, i:i+h_, :]

def _resize(im, w_, h_):
    im_resized = cv2.resize(im, (w_, h_))
    if len(im_resized.shape) == 2:
        im_resized = np.expand_dims(im_resized, -1)
    return im_resized