import json
import os
import sys
from urllib.request import urlretrieve

import mindspore as ms
from mindspore.communication import get_rank, get_local_rank

from core.models.semantic_kitti.minkunet import MinkUNet
from core.models.semantic_kitti.spvcnn import SPVCNN
from core.models.semantic_kitti.spvnas import SPVNAS

__all__ = ['spvnas_specialized', 'minkunet', 'spvcnn']


def download_url(url, model_dir='~/.torch/', overwrite=False):
    target_dir = url.split('/')[-1]
    model_dir = os.path.expanduser(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_dir = os.path.join(model_dir, target_dir)
    cached_file = model_dir
    if not os.path.exists(cached_file) or overwrite:
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        urlretrieve(url, cached_file)
    return cached_file


def spvnas_specialized(net_id, pretrained=True, **kwargs):
    url_base = 'https://hanlab.mit.edu/files/SPVNAS/spvnas_specialized/'
    net_config = json.load(
        open(
            download_url(url_base + net_id + '/net.config',
                         model_dir='.torch/spvnas_specialized/%s/' % net_id)))

    model = SPVNAS(
        net_config['num_classes'],
        macro_depth_constraint=1,
        pres=net_config['pres'],
        vres=net_config['vres'])
    model.manual_select(net_config)
    model = model.determinize()

    if pretrained:
        init = ms.load_checkpoint(download_url(url_base + net_id + '/init',
                                       model_dir='.torch/spvnas_specialized/%s/'
                                       % net_id))['model']
        model.load_state_dict(init)
    return model


def spvnas_supernet(net_id, pretrained=True, **kwargs):
    url_base = 'https://hanlab.mit.edu/files/SPVNAS/spvnas_supernet/'
    net_config = json.load(
        open(
            download_url(url_base + net_id + '/net.config',
                         model_dir='.torch/spvnas_supernet/%s/' % net_id)))

    model = SPVNAS(
        net_config['num_classes'],
        macro_depth_constraint=net_config['macro_depth_constraint'],
        pres=net_config['pres'],
        vres=net_config['vres'])

    if pretrained:
        init = ms.load_checkpoint(download_url(url_base + net_id + '/init',
                                       model_dir='.torch/spvnas_supernet/%s/'
                                       % net_id))['model']
        model.load_state_dict(init)
    return model


def minkunet(net_id, pretrained=True, **kwargs):
    url_base = 'https://hanlab.mit.edu/files/SPVNAS/minkunet/'
    net_config = json.load(
        open(
            download_url(url_base + net_id + '/net.config',
                         model_dir='.torch/minkunet/%s/' % net_id)))

    model = MinkUNet(
        num_classes=net_config['num_classes'], cr=net_config['cr'])

    if pretrained:
        init = ms.load_checkpoint(download_url(url_base + net_id + '/init',
                                       model_dir='.torch/minkunet/%s/'
                                       % net_id))['model']
        model.load_state_dict(init)
    return model


def spvcnn(net_id, pretrained=True, **kwargs):
    url_base = 'https://hanlab.mit.edu/files/SPVNAS/spvcnn/'
    net_config = json.load(
        open(
            download_url(url_base + net_id + '/net.config',
                         model_dir='.torch/spvcnn/%s/' % net_id)))

    model = SPVCNN(
        num_classes=net_config['num_classes'],
        cr=net_config['cr'],
        pres=net_config['pres'],
        vres=net_config['vres'])

    if pretrained:
        init = ms.load_checkpoint(download_url(url_base + net_id + '/init',
                                       model_dir='.torch/spvcnn/%s/' % net_id))['model']
        model.load_state_dict(init)
    return model
