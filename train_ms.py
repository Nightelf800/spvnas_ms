import argparse
import os
import time
import zipfile
import mindspore
import mindspore.dataset as ds
from mindspore import context, ParallelMode, Model, DynamicLossScaleManager
from mindspore.communication import init, get_rank, get_group_size
from core.utils.config import configs
from pip._internal import main
from src.config.option import Option
from src.dataset.perspective_view_loader import PerspectiveViewLoader
from src.dataset.semantic_kitti.parser import SemanticKitti
from src.models.pmf_net import PMFNet
from src.utils.callback import RecorderCallback, CallbackSaveByIoU
from src.utils.common import CustomWithLossCell, CustomMutiLoss, CustomMitiMomentum
from src.utils.metric import IOUEval, CustomWithEvalCell
from src.utils.local_adapter import execute_distributed


# 数据集压缩包名字，仅限Ascend，如有需要自行修改
dataset_name = "SemanticKitti.zip"


parser = argparse.ArgumentParser(description='MindSpore PMF')
parser.add_argument('--multi_data_url',
                    help='使用单数据集或多数据集时，需要定义的参数',
                    default='[{}]')

parser.add_argument('--pretrain_url',
                help='非必选，只有在界面上选择模型时才需要，使用单模型或多模型时，需要定义的参数',
                default='[{}]')
parser.add_argument('--train_url',
                    help='必选，回传结果到启智，需要定义的参数',
                    default='')
parser.add_argument("--config_path", type=str, metavar="config_path",
                    help="path of config file, type: string")
parser.add_argument(
    '--device_target',
    type=str,
    default="Ascend",
    choices=['Ascend', 'GPU'],
    help='device where the code will be implemented (default: Ascend)')


def train(settings):
    settings = settings
    mindspore.set_seed(settings.seed)

    print("-->训练使用设备: ", settings.device_target)
    if settings.device_target == "Ascend":
        device_num = int(os.getenv('RANK_SIZE'))
        rank = int(os.getenv('RANK_ID'))

        if device_num == 1:
            ###拷贝数据集到训练环境
            context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
            DatasetToEnv(args.multi_data_url, data_dir)
            pretrain_to_env(args.pretrain_url, pretrain_dir)
            zip_path = os.path.join(data_dir, dataset_name)
            zip_ref = zipfile.ZipFile(zip_path, 'r')
            zip_ref.extractall('/cache/data')
            zip_ref.close()
            is_distributed = False
        else:
            # set device_id and init for multi-card training
            context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target,
                                device_id=int(os.getenv('ASCEND_DEVICE_ID')))
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
            # Copying obs data does not need to be executed multiple times, just let the 0th card copy the data
            if rank % 8 == 0:
                ###拷贝数据集到训练环境
                DatasetToEnv(args.multi_data_url, data_dir)
                pretrain_to_env(args.pretrain_url, pretrain_dir)
                zip_path = os.path.join(data_dir, dataset_name)
                zip_ref = zipfile.ZipFile(zip_path, 'r')
                zip_ref.extractall('/cache/data')
                zip_ref.close()
                # Set a cache file to determine whether the data has been copied to obs.
                # If this file exists during multi-card training, there is no need to copy the dataset multiple times.
                f = open("/cache/download_input.txt", 'w')
                f.close()
                try:
                    if os.path.exists("/cache/download_input.txt"):
                        print("download_input succeed")
                except Exception:
                    print("download_input failed")
            while not os.path.exists("/cache/download_input.txt"):
                time.sleep(1)
            is_distributed = True
            execute_distributed()

        # 数据集路径,需要根据实际需求修改
        data_root = os.path.join(data_dir, "SemanticKitti/dataset/sequences")
        # 预训练模型路径,需要根据实际需求修改
        pretrain_path = os.path.join(pretrain_dir, "resnet34_224_revised.ckpt")
        data_config_path = "/cache/code/pmf/src/dataset/semantic_kitti/semantic-kitti.yaml"
        recorder = None

    else:
        # 分布式运行or单卡运行
        print("-->GPU数量: ", settings.n_gpus)
        rank = int(os.getenv('RANK_ID', '0'))
        if settings.n_gpus > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = settings.gpu
            context.set_context(mode=context.PYNATIVE_MODE, device_target=settings.device_target)
            init()
            rank = get_rank()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            is_distributed = True
            execute_distributed()
            if rank == 0:
                recorder = Recorder(settings, settings.save_path)
            else:
                recorder = 0
        else:
            context.set_context(mode=context.PYNATIVE_MODE, device_target=settings.device_target,
                                device_id=int(settings.gpu))
            is_distributed = False
            recorder = Recorder(settings, settings.save_path)

        data_root = settings.data_root
        pretrain_path = settings.pretrained_path
        data_config_path = "src/dataset/semantic_kitti/semantic-kitti.yaml"

    print("------------------加载模型----------------")
    # model init
    net = PMFNet(
        pcd_channels=5,
        img_channels=3,
        nclasses=settings.nclasses,
        base_channels=settings.base_channels,
        image_backbone=settings.img_backbone,
        imagenet_pretrained=settings.imagenet_pretrained,
        pretrained_path=pretrain_path
    )

    print("------------------加载数据集----------------")
    # data init
    trainset = SemanticKitti(
        root=data_root,
        sequences=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
        config_path=data_config_path
    )
    valset = SemanticKitti(
        root=data_root,
        sequences=[8],
        config_path=data_config_path
    )
    cls_weight = 1 / (valset.cls_freq + 1e-3)
    ignore_class = []
    for cl, v in enumerate(cls_weight):
        if valset.data_config["learning_ignore"][cl]:
            cls_weight[cl] = 0
        if cls_weight[cl] < 1e-10:
            ignore_class.append(cl)

    train_pv_loader = PerspectiveViewLoader(
        dataset=trainset,
        config=settings.config,
        is_train=True,
        img_aug=True,
        use_padding=True)
    val_pv_loader = PerspectiveViewLoader(
        dataset=valset,
        config=settings.config,
        is_train=False)
    if is_distributed:
        rank_size = get_group_size()
        train_loader = ds.GeneratorDataset(
            train_pv_loader,
            column_names=["pcd", "img", "mask", "label"],
            shuffle=True,
            shard_id=rank,
            num_shards=rank_size)
        val_loader = ds.GeneratorDataset(
            val_pv_loader,
            column_names=["pcd", "img", "mask", "label"],
            shuffle=False,
            shard_id=rank,
            num_shards=rank_size)
    else:
        train_loader = ds.GeneratorDataset(
            train_pv_loader,
            column_names=["pcd", "img", "mask", "label"],
            num_parallel_workers=settings.n_threads,
            python_multiprocessing=True,
            max_rowsize=32,
            shuffle=True)
        val_loader = ds.GeneratorDataset(
            val_pv_loader,
            column_names=["pcd", "img", "mask", "label"],
            num_parallel_workers=settings.n_threads,
            max_rowsize=32,
            shuffle=False)
    train_loader = train_loader.batch(
        batch_size=settings.batch_size[0],
        num_parallel_workers=settings.n_threads,
        drop_remainder=True)
    val_loader = val_loader.batch(
        batch_size=settings.batch_size[1],
        num_parallel_workers=settings.n_threads,
        drop_remainder=False)

    # metric init
    loss = CustomMutiLoss(settings, cls_weight)
    loss_net = CustomWithLossCell(settings, net, loss)
    eval_net = CustomWithEvalCell(net)
    metric = {"mIoU": IOUEval(settings.nclasses,
                              recorder=recorder,
                              ignore=ignore_class,
                              is_distributed=is_distributed)}
    opt = CustomMitiMomentum(net.lidar_stream.trainable_params(),
                             net.camera_stream_encoder.trainable_params(),
                             net.camera_stream_decoder.trainable_params(),
                             settings.lr,
                             settings.momentum,
                             settings.weight_decay,
                             train_loader.get_dataset_size(),
                             settings.n_epochs,
                             settings.warmup_epochs)

    if context.get_context('device_target') == 'Ascend':
        scale_factor = 2
        scale_window = 3000
        loss_scale_manager = DynamicLossScaleManager(scale_factor, scale_window)
        model = Model(loss_net, eval_network=eval_net, optimizer=opt, metrics=metric,
                      loss_scale_manager=loss_scale_manager)
    else:
        model = Model(loss_net, eval_network=eval_net, optimizer=opt, metrics=metric)

    recorder_cb = RecorderCallback(recorder)
    ckpoint_cb = CallbackSaveByIoU(model, val_loader, 1, 1, train_dir)
    print("------------------执行训练----------------")
    model.train(epoch=settings.n_epochs,
                train_dataset=train_loader,
                callbacks=[recorder_cb, ckpoint_cb])

    if context.get_context('device_target') == 'Ascend':
        env_to_openi(train_dir, args.train_url)


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    if args.device_target == "Ascend":
        from openi import openi_multidataset_to_env as DatasetToEnv, pretrain_to_env
        from openi import env_to_openi
        # main.main(['install', '-r', '/cache/code/pmf/requirements.txt'])
        data_dir = '/cache/data'
        train_dir = '/cache/output'
        pretrain_dir = '/cache/pretrainmodel'

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(pretrain_dir):
            os.makedirs(pretrain_dir)
    elif args.device_target == "GPU":
        from src.utils.recorder import Recorder
        train_dir = "./save_model"
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
    else:
        raise ValueError("Unsupported platform.")

    train(Option(args))
