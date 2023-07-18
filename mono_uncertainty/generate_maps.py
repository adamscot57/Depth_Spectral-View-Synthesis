#based on monodepth2
#non-commercial use only


from __future__ import absolute_import, division, print_function
import warnings

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader


from ..kitti_utils import kitti_utils
from ..layers import *
from ..utils import *
from extended_options import *
from ..datasets import datasets
from ..networks import networks as legacy
import networks
import progressbar
import matplotlib.pyplot as plt

import sys

splits_dir = os.path.join(os.path.dirname(__file__), "monodepth2/splits")

def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def get_mono_ratio(disp, gt):
    """Returns the median scaling factor
    """
    mask = gt>0
    return np.median(gt[mask]) / np.median(cv2.resize(1/disp, (gt.shape[1], gt.shape[0]))[mask])


def evaluate_model(options):
    """
    Evaluates a pretrained model using a specified test set.

    :param options: Command line arguments.
    :type options: argparse.Namespace
    :raises AssertionError: If more than one of eval_mono, eval_stereo, and no_eval are set.
    :raises AssertionError: If both log and repr are set.
    :raises AssertionError: If neither bootstraps nor snapshots are set to be greater than 1.
    """

    # Define constants
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    # Set batch size to 1
    options.batch_size = 1

    # Validate input
    assert sum((options.eval_mono, options.eval_stereo, options.no_eval)) == 1, "Please choose mono or stereo evaluation by setting either --eval_mono, --eval_stereo, --custom_run"
    assert sum((options.log, options.repr)) < 2, "Please select only one between LR and LOG by setting --repr or --log"
    assert options.bootstraps == 1 or options.snapshots == 1, "Please set only one of --bootstraps or --snapshots to be major than 1"

    # Get the number of networks
    nets = max(options.bootstraps, options.snapshots)

    # Determine whether to compute uncertainties
    do_uncert = (options.log or options.repr or options.dropout or options.post_process or options.bootstraps > 1 or options.snapshots > 1)

    logger.info("-> Beginning inference...")

    # Load weights from specified folder
    options.load_weights_folder = os.path.expanduser(options.load_weights_folder)
    assert os.path.isdir(options.load_weights_folder), "Cannot find a folder at {}".format(options.load_weights_folder)

    logger.info("-> Loading weights from {}".format(options.load_weights_folder))

    # Load test file names
    filenames = readlines(os.path.join(splits_dir, options.eval_split, "test_files.txt"))

    # Prepare dataset and dataloader
    encoder_path = os.path.join(options.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(options.load_weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path)
    height = encoder_dict['height']
    width = encoder_dict['width']
    img_ext = '.png' if options.png else '.jpg'
    dataset = datasets.KITTIRAWDataset(options.data_path, filenames, height, width, [0], 4, is_train=False, img_ext=img_ext)
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=options.num_workers, pin_memory=True, drop_last=False)

    # Load encoder and decoder
    if nets > 1:
        # Load multiple encoders and decoders
        encoder = [legacy.ResnetEncoder(options.num_layers, False) for i in range(nets)]
        depth_decoder = [networks.DepthUncertaintyDecoder(encoder[i].num_ch_enc, num_output_channels=1, uncert=(options.log or options.repr), dropout=options.dropout) for i in range(nets)]
        model_dict = [encoder[i].state_dict() for i in range(nets)]
        for i in range(nets):
            encoder[i].load_state_dict({k: v for k, v in encoder_dict[i].items() if k in model_dict[i]})
            depth_decoder[i].load_state_dict(torch.load(decoder_path[i]))
            encoder[i].cuda()
            encoder[i].eval()
            depth_decoder[i].cuda()
            depth_decoder[i].eval()
    else:
        # Load a single encoder and decoder
        encoder = legacy.ResnetEncoder(options.num_layers, False)
        depth_decoder = networks.DepthUncertaintyDecoder(encoder.num_ch_enc, num_output_channels=1, uncert=(options.log or options.repr), dropout=options.dropout)
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))
        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

    # Initialize accumulators for depth and uncertainties
    pred_disps = []
    pred_uncerts = []

    logger.info("-> Computing predictions with size {}x{}".format(width, height))
    with torch.no_grad():
        bar = progressbar.ProgressBar(max_value=len(dataloader))
        for i, data in enumerate(dataloader):

            input_color = data[("color", 0, 0)].cuda()

            # Updating progress bar
            bar.update(i)
            if options.post_process:

                # post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
            if nets > 1:

                # infer multiple predictions from multiple networks
                disps_distribution = []
                uncerts_distribution = []
                for i in range(nets):
                    output =  depth_decoder[i](encoder[i](input_color))
                    disps_distribution.append( torch.unsqueeze(output[("disp", 0)],0) )
                    if options.log:
                        uncerts_distribution.append( torch.unsqueeze( torch.exp(output[("uncert", 0)])**2, 0) )

                disps_distribution = torch.cat(disps_distribution, 0)
                if options.log:

                    # bayesian uncertainty
                    pred_uncert = torch.var(disps_distribution, dim=0, keepdim=False) +  torch.sum(torch.cat(uncerts_distribution, 0), dim=0, keepdim=False)
                else:

                    # uncertainty as variance of the predictions
                    pred_uncert = torch.var(disps_distribution, dim=0, keepdim=False)
                pred_uncert = pred_uncert.cpu()[0].numpy()
                output = torch.mean(disps_distribution, dim=0, keepdim=False)
                pred_disp, _ = disp_to_depth(output, options.min_depth, options.max_depth)
            elif options.dropout:

                # infer multiple predictions from multiple networks with dropout
                disps_distribution = []
                uncerts = []

                # we infer 8 predictions as the number of bootstraps and snaphots
                for i in range(8):
                    output = depth_decoder(encoder(input_color))
                    disps_distribution.append( torch.unsqueeze(output[("disp", 0)],0) )
                disps_distribution = torch.cat(disps_distribution, 0)

                # uncertainty as variance of the predictions
                pred_uncert = torch.var(disps_distribution, dim=0, keepdim=False).cpu()[0].numpy()

                # depth as mean of the predictions
                output = torch.mean(disps_distribution, dim=0, keepdim=False)
                pred_disp, _ = disp_to_depth(output, options.min_depth, options.max_depth)
            else:
                output = depth_decoder(encoder(input_color))
                pred_disp, _ = disp_to_depth(output[("disp", 0)], options.min_depth, options.max_depth)
                if options.log:

                    # log-likelihood maximization
                    pred_uncert = torch.exp(output[("uncert", 0)]).cpu()[:, 0].numpy()
                elif options.repr:

                    # learned reprojection
                    pred_uncert = (output[("uncert", 0)]).cpu()[:, 0].numpy()

            pred_disp = pred_disp.cpu()[:, 0].numpy()
            if options.post_process:

                # applying Monodepthv1 post-processing to improve depth and get uncertainty
                N = pred_disp.shape[0] // 2
                pred_uncert = np.abs(pred_disp[:N] - pred_disp[N:, :, ::-1])
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
                pred_uncerts.append(pred_uncert)

            pred_disps.append(pred_disp)

            # uncertainty normalization
            if options.log or options.repr or options.dropout or nets > 1:
                pred_uncert = (pred_uncert - np.min(pred_uncert)) / (np.max(pred_uncert) - np.min(pred_uncert))
                pred_uncerts.append(pred_uncert)
    pred_disps = np.concatenate(pred_disps)
    if do_uncert:
        pred_uncerts = np.concatenate(pred_uncerts)

    # saving 16 bit depth and uncertainties
    print("-> Saving 16 bit maps")
    gt_path = os.path.join(splits_dir, options.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    if not os.path.exists(os.path.join(options.output_dir, "raw", "disp")):
        os.makedirs(os.path.join(options.output_dir, "raw", "disp"))

    if not os.path.exists(os.path.join(options.output_dir, "raw", "uncert")):
        os.makedirs(os.path.join(options.output_dir, "raw", "uncert"))

    if options.qual:
        if not os.path.exists(os.path.join(options.output_dir, "qual", "disp")):
            os.makedirs(os.path.join(options.output_dir, "qual", "disp"))
        if do_uncert:
            if not os.path.exists(os.path.join(options.output_dir, "qual", "uncert")):
                os.makedirs(os.path.join(options.output_dir, "qual", "uncert"))

    bar = progressbar.ProgressBar(max_value=len(pred_disps))
    for i in range(len(pred_disps)):
        bar.update(i)

        # save uncertainties
        if do_uncert:
            cv2.imwrite(os.path.join(options.output_dir, "raw", "uncert", '%06d_10.png'%i), (pred_uncerts[i]*(256*256-1)).astype(np.uint16))

    print("\n-> Done!")

if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    options = UncertaintyOptions()
    evaluate_model(options.parse())
