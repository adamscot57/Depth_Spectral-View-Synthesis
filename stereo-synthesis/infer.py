import os
import torch
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
import sys
syn_dir = Path('./src/synHD/')
sys.path.append(str(syn_dir))

from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import src.config.test_opt as opt

#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')  ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()


visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

model = create_model(opt)


for data in tqdm(dataset):
    print('test data: ', data)
    minibatch = 1
    generated = model.inference(data['label'], data['inst'])

    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    visualizer.save_images(webpage, visuals, img_path)
webpage.save()
torch.cuda.empty_cache()
