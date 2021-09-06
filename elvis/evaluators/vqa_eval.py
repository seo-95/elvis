import argparse
import json
import os
import pdb

import numpy as np
import torch
import yaml
from detectron2.data.transforms import ResizeShortestEdge
from elvis.config import ConfigNode
from elvis.modeling import build_meta_vqa
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# MAKE IT CALLABLE DIRECTLY VIA `python vqa_eval.py --args`


# PSEUDO CODE:
#   load from checkpoint folder (cfg, model parameters, answers vocabulary)
#   iterate over VQA test set
#       save answers in natural language (id2ans) inside a list
#   call the evaluator from VQA library


def move_input_to_cuda(input_data):
    for k in input_data.keys():
        if isinstance(input_data[k], torch.Tensor,):
            input_data[k] = input_data[k].cuda()

def prepare_img(img_path, transforms, limit_edge=None):
    #open image and do transformations
    img = Image.open(img_path).convert('RGB')
    img = transforms(img)
    if limit_edge is not None:
        #randaugment works with PIL. ResizeShortest works with numpy.ndarray
        img = np.array(img.permute(1, 2, 0))
        img = limit_edge.get_transform(img).apply_image(img)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) #from float64 to float32 to allow mixed precision
    return img

def evaluate_vqa(model, data, data_interface, img_dir, cuda=False, **kwargs):
    res_dict = []
    if cuda:
        model.cuda()
    for item in tqdm(data['questions']):
        img_path    = os.path.join(img_dir, '{}.jpg'.format(item['image_id']))
        item['img'] = prepare_img(img_path, **kwargs)
        input_data  = data_interface.worker_fn_eval(item)
        if cuda:
            move_input_to_cuda(input_data)
        answer, ans_conf   = model.predict(**input_data)
        print(answer, ans_conf)
        res_dict.append({'question_id': item['question_id'], 'answer': answer})
    return res_dict, data['data_subtype']



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        help='Directory where to find the model to evaluate (configuration file, parameters and answers vocabulary)'
    )
    parser.add_argument(
        '--questions_file',
        type=str,
        help='JSON file containing the questions'
    )
    parser.add_argument(
        '--img_dir',
        type=str,
        help='Directory where to find the images'
    )
    parser.add_argument(
        '--cuda',
        action='store_true',
        required=False,
        default=False,
        help='Flag to use GPU'
    )

    args = parser.parse_args()
    cfg_file = os.path.join(args.model_dir, 'cfg.yaml')
    with open(cfg_file) as fp:
        cfg = yaml.safe_load(fp)
        cfg = ConfigNode(cfg)
    cfg.MODEL.ANS_VOCAB = os.path.join(args.model_dir, 'VQA.vocab')

    with open(args.questions_file) as fp:
        data = json.load(fp)

    tr = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(cfg.TRAINER.DATASET.MEAN, cfg.TRAINER.DATASET.STD)])
    #Resize before RandAugment
    if cfg.TRAINER.DATASET.has_attr('RESIZE'):
        tr.transforms.insert(0, transforms.Resize(size=cfg.TRAINER.DATASET.RESIZE))
    limit_edge = ResizeShortestEdge(*cfg.TRAINER.DATASET.LIMIT_EDGE) if cfg.TRAINER.DATASET.has_attr('LIMIT_EDGE') else None
    kwargs = {'transforms': tr, 'limit_edge': limit_edge}

    model, data_interface = build_meta_vqa(cfg)
    data_interface.eval_mode = True
    params_file = os.path.join(args.model_dir, 'state_dict.pt')
    state_dict  = torch.load(params_file)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        res, name = evaluate_vqa(model, data, data_interface, args.img_dir, cuda=args.cuda, **kwargs)
    res_file = os.path.join(args.model_dir, '{}_results.txt'.format(name))
    with open(res_file, 'w') as fp:
        json.dump(res)
    print('Evaluation completed!')

    
