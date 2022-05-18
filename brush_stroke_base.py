import os
import sys
import torch
import torch as T
import torch.optim as optim
from torchvision.transforms import functional as F
from neural_monitor import monitor as mon
from neural_monitor import logger
import argparse

from param_stroke import BrushStrokeRenderer
import utils
import losses
import get_contours
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('content_img_file', type=str, help='Content image file')
parser.add_argument('style_img_file', type=str, help='Style image file')
parser.add_argument('--img_size', '-s', type=int, default=256,
                    help='The smaller dimension of content image is resized into this size. Default: 512.')
parser.add_argument('--canvas_color', default='gray', type=str,
                    help='Canvas background color (`gray` (default), `white`, `black` or `noise`).')
parser.add_argument('--num_strokes', default=5000, type=int,
                    help='Number of strokes to draw. Default: 5000.')
parser.add_argument('--samples_per_curve', default=30, type=int,
                    help='Number of points to sample per parametrized curve. Default: 10.')
parser.add_argument('--brushes_per_pixel', default=20, type=int,
                    help='Number of brush strokes to be drawn per pixel. Default: 20.')
parser.add_argument('--output_path', '-o', type=str, default='results',
                    help='Storage for results. Default: `results`.')
parser.add_argument('--device', '-d', type=str, default='cuda',
                    help='Device to perform stylization. Default: `cuda`.')
parser.add_argument('--save_to', type=str, default='./temp_image.jpg',
                    help='Where to save output. Default: temp_image, Not saving: None')
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# inputs
style_img_file = args.style_img_file
content_img_file = args.content_img_file

# setup logging
model_name = 'nst-stroke'
root = args.output_path
vgg_weight_file = 'vgg_weights/vgg19_weights_normalized.h5'
print_freq = 10
mon.initialize(model_name=model_name, root=root, print_freq=print_freq)
mon.backup(('main.py', 'param_stroke.py', 'utils.py', 'losses.py', 'vgg.py'))

# device
device = torch.device(args.device)

# desired size of the output image
imsize = args.img_size
content_img = utils.image_loader(content_img_file, imsize, device)
style_img = utils.image_loader(style_img_file, 224, device)
output_name = f'{os.path.basename(content_img_file).split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}'

# desired depth layers to compute style/content losses :
bs_content_layers =  ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'] #['conv4_1', 'conv5_1'] 
bs_style_layers = ['conv4_1', 'conv5_1']  #['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']


# brush strokes parameters
canvas_color = args.canvas_color
num_strokes = args.num_strokes
samples_per_curve = args.samples_per_curve
brushes_per_pixel = args.brushes_per_pixel
_, _, H, W = content_img.shape
canvas_height = H
canvas_width = W
length_scale = 2.1 #1.1
width_scale = 90.5 #0.1

a_min, a_max = get_contours.get_ellipse_stats(style_img, n_iters = 1000, need_graphics = False)




def run_stroke_style_transfer(num_steps=1000, style_weight=3., content_weight=1., tv_weight=0.008, 
                              curv_weight=4, len_weight = 0.001, wid_weight = 0.0001, init='grid', name = ''):
                              
    vgg_loss = losses.StyleTransferLosses(vgg_weight_file, content_img, style_img,
                                          bs_content_layers, bs_style_layers, scale_by_y=True)
    vgg_loss.to(device).eval()

    # brush stroke init
    bs_renderer = BrushStrokeRenderer(canvas_height, canvas_width, num_strokes, samples_per_curve, brushes_per_pixel,
                                      canvas_color, length_scale, width_scale,
                                      content_img=content_img[0].permute(1, 2, 0).cpu().numpy(), 
                                      style_image=args.style_img_file, init=init)

    bs_renderer.to(device)

    optimizer = optim.Adam([bs_renderer.location, bs_renderer.curve_s, bs_renderer.curve_e,
                            bs_renderer.curve_c, bs_renderer.width], lr=1e-1)


    logger.info('Optimizing brushstroke-styled canvas..')
    for _ in mon.iter_batch(range(num_steps)):
        optimizer.zero_grad()

        input_img = bs_renderer()
        input_img = input_img[None].permute(0, 3, 1, 2).contiguous()
        content_score, style_score = vgg_loss(input_img)

        style_score *= style_weight
        content_score *= content_weight
        tv_score = tv_weight * losses.total_variation_loss(bs_renderer.location, bs_renderer.curve_s,
                                                           bs_renderer.curve_e, K=10)

        curv_score = curv_weight * losses.curvature_loss(bs_renderer.curve_s, bs_renderer.curve_e, bs_renderer.curve_c)
        length_score = len_weight * losses.length_loss(bs_renderer.curve_s, bs_renderer.curve_e, bs_renderer.curve_c, a_max // 6)
        
        width_score = wid_weight * losses.width_loss(bs_renderer.width, a_min // 4)
        
        loss = style_score + content_score +  curv_score + tv_score + length_score + width_score
        
        loss.backward(inputs=[bs_renderer.location, bs_renderer.curve_s, bs_renderer.curve_e,
                              bs_renderer.curve_c, bs_renderer.width], retain_graph=True)
        optimizer.step()

        mon.plot('stroke style loss'+ '_' + str(len_weight) + '_' + str(wid_weight) + '_', style_score.item())
        mon.plot('stroke content loss'+ '_' + str(len_weight) + '_' + str(wid_weight) + '_', content_score.item())
        mon.plot('stroke tv loss'+ '_' + str(len_weight) + '_' + str(wid_weight) + '_', tv_score.item())
        mon.plot('stroke curvature loss'+ '_' + str(len_weight) + '_' + str(wid_weight) + '_', curv_score.item())

    mon.imwrite('stroke stylized' + '_' + str(len_weight) + '_' + str(wid_weight) + '_', input_img)

    with T.no_grad():
        return bs_renderer()






if __name__ == '__main__':

    

    
    
    mon.iter = 0
    mon.print_freq = 100

    
    canvas = run_stroke_style_transfer(wid_weight=0.000001, len_weight=0.1, style_weight=3.0, content_weight=1.0, init='grid', name = '')
    

    save_image(style_img.cpu(), args.save_to)
    
    