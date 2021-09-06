import argparse
import json
import os
import pdb

import numpy as np
import torch
import yaml
from detectron2.data.transforms import ResizeShortestEdge
from elvis.config import ConfigNode
from elvis.modeling import build_meta_retrieval
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# MAKE IT CALLABLE DIRECTLY VIA `python retrieval_eval.py --args`


# PSEUDO CODE:
#   load from checkpoint (cfg, model parameters)
#   iterate over MSCOCO 1k/5k
#       for each row just save the position assigned to the true pair
#   save list of positions on a file
#   compute R@1, R@5, R@10

# estimated time with no batching and no images pre-loading: 327h (12.5 days per fold)
# estimated time with images pre-loading: 70h (3 days per fold)
# estimated time with images pre-loading + images batching: 58h (2.4 days per fold -> 12 days for 5 folds)


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

def compute_recalls(ranks):
    results = {'recall@1': 0, 'recall@5': 0, 'recall@10': 0}
    for r in ranks:
        results['recall@1']  += 1 if r <= 1 else 0
        results['recall@5']  += 1 if r <= 5 else 0
        results['recall@10'] += 1 if r <= 10 else 0
    for k in results.keys():
        results[k] /= len(ranks)
    return results

def evaluate_retrieval(model, data, data_interface, img_dir, cuda=False, **kwargs):
    print('Loading images into memory ...')
    img_dict = {}
    for img_id in tqdm(data['images']):
        img_path = os.path.join(img_dir, '{}.jpg'.format(img_id))
        img      = prepare_img(img_path, **kwargs)
        img_dict[img_id] = img
    ranks = []
    for item in tqdm(data['annotations']):
        rank       = 1
        imgs       = [img_dict[img_id] for img_id in item['pool']]

        #no more than 500 images fit on 8GB GPU. So divide the pool in 2 halfs
        input_half_1 = {'id': item['id'], 'caption': item['caption'], 'imgs': imgs[:500], 'image_ids': item['pool']}
        input_half_1 = data_interface.worker_fn_eval(input_half_1)
        scores_half_1 = model.predict(**input_half_1)
        del input_half_1
        torch.cuda.empty_cache()

        input_half_2 = {'id': item['id'], 'caption': item['caption'], 'imgs': imgs[500:], 'image_ids': item['pool']}
        input_half_2 = data_interface.worker_fn_eval(input_half_2)
        scores_half_2 = model.predict(**input_half_2)
        del input_half_2
        torch.cuda.empty_cache()

        scores = scores_half_1 + scores_half_2
        true_image_score = scores[0]
        rank = 1
        for score in scores[1:]:
            rank += 1 if score > true_image_score else 0
        ranks.append(rank)

    recalls = compute_recalls(ranks)
    return recalls, ranks


def remove_mlm_layers(state_dict):
    layers_names = list(state_dict.keys())
    for l_name in layers_names:
        if l_name.startswith('lm_mlp'):
            del state_dict[l_name]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        help='Directory where to find the model to evaluate (configuration file, parameters and answers vocabulary)'
    )
    parser.add_argument(
        '--annotations_file',
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

    with open(args.annotations_file) as fp:
        data = json.load(fp)


    tr = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(cfg.TRAINER.DATASET.MEAN, cfg.TRAINER.DATASET.STD)])
    #Resize before RandAugment
    if cfg.TRAINER.DATASET.has_attr('RESIZE'):
        tr.transforms.insert(0, transforms.Resize(size=cfg.TRAINER.DATASET.RESIZE))
    limit_edge = ResizeShortestEdge(*cfg.TRAINER.DATASET.LIMIT_EDGE) if cfg.TRAINER.DATASET.has_attr('LIMIT_EDGE') else None
    kwargs = {'transforms': tr, 'limit_edge': limit_edge}

    cfg.MODEL.MAX_N_TOKENS = 16-2 #TODO REMOVE
    cfg.MODEL.MAX_VIS_PATCHES  = 200-1 #TODO REMOVE

    model, data_interface = build_meta_retrieval(cfg)
    data_interface.eval_mode = True
    params_file = os.path.join(args.model_dir, 'state_dict.pt')
    state_dict  = torch.load(params_file)
    if cfg.MODEL.NAME == 'build_align_vlp':
        remove_mlm_layers(state_dict)

    model.load_state_dict(state_dict)
    model.eval()
    model.cuda() #always use cuda (otherwise unfeasible)
    with torch.no_grad():
        recalls, ranks = evaluate_retrieval(model, data, data_interface, args.img_dir, cuda=args.cuda, **kwargs)
    res = {'results': recalls, 'ranks': ranks}

    fold = args.annotations_file.split('/')[-1].split('.')[0].split('_')[-1]
    res_file = os.path.join(args.model_dir, 'retrieval-1k-{}.json'.format(fold))
    with open(res_file, 'w') as fp:
        json.dump(res, fp)
    print('Recalls: {}'.format(recalls))
    print('Evaluation completed! Results saved in {}'.format(res_file))



