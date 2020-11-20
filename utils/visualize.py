"""
Visualizations for input and output tensors
"""

from matplotlib import pyplot as plt

def tensor2double(inp, segmentation_ch=False,
              mean_rgb=[0.485, 0.456, 0.406], std_rgb=[0.229, 0.224, 0.225],
              mean_seg=[0.5], std_seg=[0.5]):
    """
    Convert the tensor image into numpy array in the range 0,1
    Assumes batch_size=1

    inp (Tensor of size (CxWxH)): Input tensor of size 1xCxWxH
    
    returns:
    (np array iof size (WxHXC)): output array
    """

    inp_numpy = inp.cpu().numpy()[0, :, :, :].transpose(1, 2, 0)
    w, h, c = inp_numpy.shape

    num_cahnnels_per_fr = 3+(1*segmentation_ch)
    num_frames = int(c/num_cahnnels_per_fr) # empty bg (?), recent bg (?), current fr


    for ch in range(num_frames):
        im = inp_numpy[:, :, num_cahnnels_per_fr*ch+(1*segmentation_ch):num_cahnnels_per_fr*(ch+1)]
        im = (im*std_rgb)+mean_rgb
        inp_numpy[:, :, num_cahnnels_per_fr*ch+(1*segmentation_ch):num_cahnnels_per_fr*(ch+1)] = im
        if segmentation_ch:
            im = inp_numpy[:, :, num_cahnnels_per_fr*ch]
            im = (im*std_seg) + std_seg
            inp_numpy[:, :, num_cahnnels_per_fr*ch] = im

    return inp_numpy

def visualize(inp, out, segmentation_ch=False,
              mean_rgb=[0.485, 0.456, 0.406], std_rgb=[0.229, 0.224, 0.225],
              mean_seg=[0.5], std_seg=[0.5]):
    """
    Shows the first input and output data from the minibatch in matplotlib

    inp (Tensor of size (CxWxH)): Input tensor of size BxCxWxH
    out (Tensor of size (1xWxH)): Output tensor of size Bx1xWxH
    """

    inp_numpy = inp.cpu().numpy()[0, :, :, :].transpose(1, 2, 0)
    out_numpy = out.cpu().numpy()[0, 0, :, :]
    w, h, c = inp_numpy.shape

    num_cahnnels_per_fr = 3+(1*segmentation_ch)
    num_frames = int(c/num_cahnnels_per_fr) # empty bg (?), recent bg (?), current fr

    fig, axes = plt.subplots((1+(1*segmentation_ch)), num_frames+1, figsize=(40, 20))

    im_arr = []
    for ch in range(num_frames):
        im = inp_numpy[:, :, num_cahnnels_per_fr*ch+(1*segmentation_ch):num_cahnnels_per_fr*(ch+1)]
        im = (im*std_rgb)+mean_rgb
        im_arr.append(im)
        axes[0, ch].imshow(im)
        if segmentation_ch:
            im = inp_numpy[:, :, num_cahnnels_per_fr*ch]
            im = (im*std_seg) + std_seg
            axes[1, ch].imshow(im)

    axes[0, num_frames].imshow(out_numpy)
    im_arr.append(out_numpy)
    return im_arr