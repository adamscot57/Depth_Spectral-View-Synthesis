import os
import numpy as np
import torch
import time
import sys
from collections import OrderedDict
from torch.autograd import Variable
from pathlib import Path
import pytorch3d
import warnings
import logging

warnings.filterwarnings('ignore')
mainpath = os.getcwd()
syn_dir = Path(mainpath+'/src/synHD/')
sys.path.append(str(syn_dir))

from tqdm import tqdm
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import src.config.train_opt as opt

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(opt):
    """
    Trains the model using the specified options.

    :param opt: Options for training the model.
    :type opt: argparse.Namespace
    """
    
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    
    # Check if training should be resumed from a checkpoint
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            logger.error("Could not load iteration file. Starting from beginning.")
            start_epoch, epoch_iter = 1, 0
        logger.info('Resuming from epoch %d at iteration %d', start_epoch, epoch_iter)
    else:
        start_epoch, epoch_iter = 1, 0
        logger.info('Starting from beginning')
    
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    logger.info('#training images = %d', dataset_size)
    
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq
    
    # Create model and visualizer
    model = create_model(opt)
    model = model.cuda()
    visualizer = Visualizer(opt)
    
    # Train model for specified number of epochs
    for epoch in range(start_epoch, opt.niter + opt.niter_decay +1):
        epoch_start_time = time.time()
        
        # Reset epoch iterator if not first epoch
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        
        # Train model on each batch of data in the dataset
        for i, data in tqdm(enumerate(dataset, start=epoch_iter)):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            
            save_fake = total_steps % opt.display_freq == display_delta
            
            losses, generated = model(Variable(data['label']), Variable(data['inst']),
                                      Variable(data['image']), Variable(data['feat']), infer=save_fake)
            
            # Sum per device losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.loss_names, losses))
            
            # Calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)
            
            # Backward pass and update weights
            model.optimizer_G.zero_grad()
            loss_G.backward()
            model.optimizer_G.step()
            
            model.optimizer_D.zero_grad()
            loss_D.backward()
            model.optimizer_D.step()
            
            # Display results and errors
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)
            
            if save_fake:
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                       ('synthesized_image', util.tensor2im(generated.data[0])),
                                       ('real_image', util.tensor2im(data['image'][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)
            
            # Save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                logger.info('Saving the latest model (epoch %d, total_steps %d)', epoch, total_steps)
                model.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            
            # End epoch if all batches have been trained on
            if epoch_iter >= dataset_size:
                break
        
        # End of epoch
        logger.info('End of epoch %d / %d \t Time Taken: %d sec', epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time)
        
        # Save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            logger.info('Saving the model at the end of epoch %d, iters %d', epoch, total_steps)
            model.save('latest')
            model.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')
        
        # Train entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.update_fixed_params()
        
        # Linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.update_learning_rate()
    
    # Clear GPU memory
    torch.cuda.empty_cache()

if __name__ == '__main__':
    opt = get_options()
    train(opt)