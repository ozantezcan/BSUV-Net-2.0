#Imports

import argparse
import os
import configs.data_config as data_config
import configs.full_cv_config as tr_test_config
import torch
import torch.optim as optim
from utils import augmentations as aug
from utils.data_loader import CDNet2014Loader
from utils import losses
from models.unet import unet_vgg16
from utils.eval_utils import logVideos
from tensorboardX import SummaryWriter
import time


parser = argparse.ArgumentParser(description='BSUV-Net-2.0 pyTorch')
parser.add_argument('--network', metavar='Network', dest='network', type=str, default='unetvgg16',
                    help='Which network to use')

# Input images
parser.add_argument('--inp_size', metavar='Input Size', dest='inp_size', type=int, default=224,
                    help='Size of the inputs. If equals 0, use the original sized images. Assumes square sized input')
parser.add_argument('--empty_bg', metavar='Empty Background Frame', dest='empty_bg', type=str, default='manual',
                    help='Which empty background to use? no, manual or automatic')
parser.add_argument('--recent_bg', metavar='Recent Background Frame', dest='recent_bg', type=int, default=1,
                    help='Use recent background frame as an input as well. 0 or 1')
parser.add_argument('--seg_ch', metavar='Segmentation', dest='seg_ch', type=int, default=1,
                    help='Whether to use the FPM channel input or not. 0 or 1')

# Optimization
parser.add_argument('--lr', metavar='Learning Rate', dest='lr', type=float, default=1e-4,
                    help='learning rate of the optimization')
parser.add_argument('--weight_decay', metavar='weight_decay', dest='weight_decay', type=float, default=1e-2,
                    help='weight decay of the optimization')
parser.add_argument('--num_epochs', metavar='Number of epochs', dest='num_epochs', type=int, default=200,
                    help='Maximum number of epochs')
parser.add_argument('--batch_size', metavar='Minibatch size', dest='batch_size', type=int, default=4,
                    help='Number of samples per minibatch')
parser.add_argument('--loss', metavar='Loss function to be used', dest='loss', type=str, default='jaccard',
                    help='Loss function to be used ce for cross-entropy or jaccard for Jaccard index')
parser.add_argument('--opt', metavar='Optimizer to be used', dest='opt', type=str, default='adam',
                    help='sgd, rmsprop or adam')

# Data augmentations
parser.add_argument('--aug_noise', metavar='Data Augmentation for noise', dest='aug_noise', type=int, default=1,
                    help='Whether to use Data Augmentation for noise. 0 or 1')
parser.add_argument('--aug_rsc', metavar='Data Augmentation for randomly-shifted crop', dest='aug_rsc', type=int, default=1,
                    help='Whether to use randomly-shifted crop. 0 or 1')
parser.add_argument('--aug_ptz', metavar='Data Augmentation for PTZ camera crop', dest='aug_ptz', type=int, default=1,
                    help='Whether to use PTZ camera crop 0 or 1')
parser.add_argument('--aug_id', metavar='Data Augmentation for Illumination Difference', dest='aug_id', type=int, default=1,
                    help='Whether to use Data Augmentation for Illumination Difference. 0 or 1')
parser.add_argument('--aug_ioa', metavar='Data Augmentation for Intermittent Object Addition', dest='aug_ioa', type=float, default=0.1,
                    help='Probability of applying Intermittent Object Addition')

# Checkpoint
parser.add_argument('--model_chk', metavar='Checkpoint for the model', dest='model_chk', type=str, default='',
                    help='Date of the checkpoint model if there exist any')

# Cross-validation
parser.add_argument('--set_number', metavar='Which training-test split to use from config file', dest='set_number',
                     type=int, default=1, help='Training and test videos will be selected based on the set number')

# Model name
parser.add_argument('--model_name', metavar='Name of the model for log keeping', dest='model_name',
                     type=str, default='BSUV-Net 2.0', help='Name of the model to be used in output csv and checkpoints')

args = parser.parse_args()
network = args.network
empty_bg = args.empty_bg
recent_bg = True if args.recent_bg == 1 else False
seg_ch = True if args.seg_ch == 1 else False
lr = args.lr
weight_decay = args.weight_decay
num_epochs = args.num_epochs
batch_size = args.batch_size
loss = args.loss
opt = args.opt
inp_size = args.inp_size
if inp_size == 0:
    inp_size = None
else:
    inp_size = (inp_size, inp_size)

aug_noise = args.aug_noise
aug_rsc = args.aug_rsc
aug_ptz = args.aug_ptz
aug_id = args.aug_id
aug_ioa = args.aug_ioa

model_chk = args.model_chk
if model_chk == '':
    model_chk = None
set_number = args.set_number
cuda = True

# Initializations
dataset_tr = tr_test_config.datasets_tr[set_number]
dataset_test = tr_test_config.datasets_test[set_number]
save_dir = data_config.save_dir

# naming for log keeping
if model_chk:
    fname = model_chk
else:
    fname = args.model_name

print(f"Model started: {fname}")

# Intialize Tensorboard
writer = SummaryWriter("tb_runs/{}".format(fname))
print("Initialized TB")

mdl_dir = os.path.join(save_dir, fname)

if not os.path.exists(mdl_dir):
    os.makedirs(mdl_dir)

# Augmentations
crop_and_aug = [aug.RandomCrop(inp_size)]

if aug_rsc:
    crop_and_aug.append(aug.RandomJitteredCrop(inp_size))
    
if aug_ptz > 0:
    crop_and_aug.append(
        [
            aug.RandomZoomCrop(inp_size),
            aug.RandomPanCrop(inp_size),
        ]
    )

additional_augs = []

if aug_id:
    ill_global, std_ill_diff = (0.1, 0.04), (0.1, 0.04)
    additional_augs.append([aug.AdditiveRandomIllumation(ill_global, std_ill_diff)])

if aug_ioa > 0:
    additional_augs_iom = []
    if aug_id:
        ill_global, std_ill_diff = (0.1, 0.04), (0.1, 0.04)
        additional_augs_iom.append([aug.AdditiveRandomIllumation(ill_global, std_ill_diff)])

    if aug_noise:
        noise = 0.01
        additional_augs_iom.append([aug.AdditiveNoise(noise)])

    iom_dataset = {
        'intermittentObjectMotion':dataset_tr['intermittentObjectMotion']
    }
    mask_transforms = [
        [aug.RandomCrop(inp_size)],
        *additional_augs_iom,
    ]

    dataloader_mask = CDNet2014Loader(
        iom_dataset, 
        empty_bg=empty_bg, 
        recent_bg=recent_bg,
        segmentation_ch=seg_ch,
        transforms=mask_transforms,
        multiplier=0,
        shuffle=True
        )

    additional_augs.append([aug.RandomMask(inp_size, dataloader_mask, mask_prob=aug_ioa)])

if aug_noise:
    noise = 0.01
    additional_augs.append([aug.AdditiveNoise(noise)])

mean_rgb = [x for x in [0.485, 0.456, 0.406]]
std_rgb = [x for x in [0.229, 0.224, 0.225]]
mean_seg = [x for x in [0.5]]
std_seg = [x for x in [0.5]]

transforms_tr = [
    crop_and_aug,
    *additional_augs,
    [aug.ToTensor()],
    [aug.NormalizeTensor(mean_rgb=mean_rgb, std_rgb=std_rgb,
                        mean_seg=mean_seg, std_seg=std_seg, segmentation_ch=seg_ch)]
]

transforms_test = [
    [aug.CenterCrop(inp_size)],
    [aug.ToTensor()],
    [aug.NormalizeTensor(mean_rgb=mean_rgb, std_rgb=std_rgb,
                        mean_seg=mean_seg, std_seg=std_seg, segmentation_ch=seg_ch)]
]


dataloader_tr = CDNet2014Loader(
                    dataset_tr, empty_bg=empty_bg, recent_bg=recent_bg,
                    segmentation_ch=seg_ch, transforms=transforms_tr,
)
dataloader_test = CDNet2014Loader(
                    dataset_test, empty_bg=empty_bg, recent_bg=recent_bg,
                    segmentation_ch=seg_ch, transforms=transforms_test,
)

tensorloader_tr = torch.utils.data.DataLoader(
    dataset=dataloader_tr, batch_size=batch_size, shuffle=True, num_workers=1
)
tensorloader_test = torch.utils.data.DataLoader(
    dataset=dataloader_test, batch_size=batch_size, shuffle=False, num_workers=1
)

# load model
num_ch_per_inp = (3 + (1*seg_ch))
num_inp = ((1*(empty_bg != "no")) + (1*recent_bg) + 1)
num_ch = num_inp * num_ch_per_inp

if network == "unetvgg16":
    model = unet_vgg16(inp_ch=num_ch, skip=1)
else:
    raise ValueError("network = {} is not defined".format(network))

for p in model.parameters():
    p.requires_grad = True

# setup optimizer
if opt == "adam":
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
elif opt == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
else:
    raise("opt=%s is not defined, please choose from ('adam', 'sgd')." %opt)

if loss == "jaccard":
    loss_func = losses.jaccard_loss
else:
    raise ("loss=%s is not defined, please choose from ('jaccard')." % opt)

if cuda:
    model = model.cuda()

if model_chk:
    chk_path = "{}/checkpoint.pth".format(mdl_dir)
    assert os.path.isfile(chk_path), "No checkpoint is found in {}".format(chk_path)
    print("=> loading checkpoint '{}'".format(model_chk))
    checkpoint = torch.load(chk_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(model_chk, checkpoint['epoch']))
else:
    start_epoch = 0

# training
best_f = 0.0 # For saving the best model
st = time.time()
for epoch in range(start_epoch, num_epochs):  # loop over the dataset multiple times
    for phase, tensorloader in [("Train", tensorloader_tr), ("Test", tensorloader_test)]:
        running_loss, running_acc, running_f = 0.0, 0.0, 0.0
        if phase == "Train":
            optimizer.zero_grad()
        for i, data in enumerate(tensorloader):
            if phase == "Train":
                model.train()
            else:
                model.eval()

            if phase == "Train":
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].float(), data[1].float()
                if cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                labels_1d, outputs_1d = losses.getValid(labels, outputs)
                loss = loss_func(labels_1d, outputs_1d)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                with torch.no_grad():
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data[0].float(), data[1].float()
                    if cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    labels_1d, outputs_1d = losses.getValid(labels, outputs)
                    loss = loss_func(labels_1d, outputs_1d)

            # print statistics
            running_loss += loss.item()
            running_acc += losses.acc(labels_1d, outputs_1d).item()
            running_f += losses.f_score(labels_1d, outputs_1d).item()

            del inputs, labels, outputs, labels_1d, outputs_1d
            if (i+1) % 10000 == 0:    # print every 2000 mini-batches
                print("::%s::[%d, %5d] loss: %.1f, acc: %.3f, f_score: %.3f" %
                      (phase, epoch + 1, i + 1,
                       running_loss / (i+1), running_acc / (i+1), running_f / (i+1)))

        epoch_loss = running_loss / len(tensorloader)
        epoch_acc = running_acc / len(tensorloader)
        epoch_f = running_f / len(tensorloader)

        current_lr = lr
        print("::%s:: Epoch %d loss: %.1f, acc: %.3f, f_score: %.3f, lr x 1000: %.4f, elapsed time: %s" \
              % (phase, epoch + 1, epoch_loss, epoch_acc, epoch_f, current_lr*1000, (time.time() - st)))

        writer.add_scalar("{}/loss".format(phase), epoch_loss, epoch)
        writer.add_scalar("{}/acc".format(phase), epoch_acc, epoch)
        writer.add_scalar("{}/f".format(phase), epoch_f, epoch)

        if phase.startswith("Test"):
            best_f = epoch_f
            torch.save(model, "{}/model_best.mdl".format(mdl_dir))

        # Save the checkpoint
        checkpoint = {
            "epoch": epoch+1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        torch.save(checkpoint, "{}/checkpoint.pth".format(mdl_dir))  
        if (epoch + 1) % 20 == 0:
            torch.save(model, "{}/model_epoch{}.mdl".format(mdl_dir, epoch + 1))

        st = time.time()

# save the last model
torch.save(model, "{}/model_last.mdl".format(mdl_dir))

print('Finished Training')

# Evaluation on test videos
model = torch.load("{}/model_best.mdl".format(mdl_dir)).cuda()
csv_path = "./log.csv"
logVideos(
    dataset_test, 
    model, 
    fname, 
    csv_path,
    empty_bg=empty_bg, 
    recent_bg=recent_bg,
    segmentation_ch=seg_ch, 
    save_vid=0, 
    set_number=set_number, 
    debug=0
)
print(f"Saved results to {csv_path}")
