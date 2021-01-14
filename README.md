# Maximum Density Divergence for Domain Adaptation (MDD)

This is a simple pytorch-lightning reimplementation of the 'Maximum Density Divergence for Domain Adaptation' paper, which can be read [here](https://arxiv.org/pdf/2004.12615.pdf), while the code has been adapted from [here](https://github.com/lijin118/ATM). 

## Maximum Density Divergence

As stated in the paper abstract:
"*Unsupervised domain adaptation addresses the problem of transferring knowledge from a well-labeled source domain to an unlabeled target domain where the two domains have distinctive data distributions. Thus, the essence of domain adaptation is to mitigate the distribution divergence between the two domains. The state-of-the-art methods practice this very idea by either conducting adversarial training or minimizing a metric which defines the distribution gaps. In this paper, we propose a new domain adaptation method named Adversarial Tight Match (ATM) which enjoys the benefits of both adversarial training and metric learning. Specifically, at first, we propose a novel distance loss, named Maximum Density Divergence (MDD), to quantify the distribution divergence. MDD minimizes the inter-domain divergence and maximizes the intra-class density. Then, to address the equilibrium challenge issue in  adversarial domain adaptation, we consider leveraging the proposed MDD into adversarial domain adaptation framework*".  

The general framework is depicted in the following figure:

![MDD](imgs/atm.jpg)

# Download datasets

Before training, one can automatically download [Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code) and [Image-clef](https://www.imageclef.org/2014/adaptation) datasets with the following commands:

* `python download_dataset.py --dataset office-31`
* `python download_dataset.py --dataset image-clef`

Soon will be availble other datasets, such as MNIST, SVHN and so on.

# Requirements

This repo has been tested on a Linux machine (Ubuntu 18.04 LTS) with:
* python 3.8
* [Albumentations](https://albumentations.ai/)
* [PyTorch](https://pytorch.org/) 1.7
* [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) 1.1.3

To completely install the requirements run:

* `pip install -U -r requirements.txt`

# Command line arguments

To train a model one have to execute the `main.py` script, that can be run with the following command line arguments:

```
usage: main.py [-h] [--logger [LOGGER]] [--checkpoint_callback [CHECKPOINT_CALLBACK]] [--default_root_dir DEFAULT_ROOT_DIR]
               [--gradient_clip_val GRADIENT_CLIP_VAL] [--process_position PROCESS_POSITION] [--num_nodes NUM_NODES] [--num_processes NUM_PROCESSES]
               [--gpus GPUS] [--auto_select_gpus [AUTO_SELECT_GPUS]] [--tpu_cores TPU_CORES] [--log_gpu_memory LOG_GPU_MEMORY]
               [--progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE] [--overfit_batches OVERFIT_BATCHES] [--track_grad_norm TRACK_GRAD_NORM]
               [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--fast_dev_run [FAST_DEV_RUN]] [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES]
               [--max_epochs MAX_EPOCHS] [--min_epochs MIN_EPOCHS] [--max_steps MAX_STEPS] [--min_steps MIN_STEPS]
               [--limit_train_batches LIMIT_TRAIN_BATCHES] [--limit_val_batches LIMIT_VAL_BATCHES] [--limit_test_batches LIMIT_TEST_BATCHES]
               [--val_check_interval VAL_CHECK_INTERVAL] [--flush_logs_every_n_steps FLUSH_LOGS_EVERY_N_STEPS] [--log_every_n_steps LOG_EVERY_N_STEPS]
               [--accelerator ACCELERATOR] [--sync_batchnorm [SYNC_BATCHNORM]] [--precision PRECISION] [--weights_summary WEIGHTS_SUMMARY]
               [--weights_save_path WEIGHTS_SAVE_PATH] [--num_sanity_val_steps NUM_SANITY_VAL_STEPS] [--truncated_bptt_steps TRUNCATED_BPTT_STEPS]
               [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--profiler [PROFILER]] [--benchmark [BENCHMARK]] [--deterministic [DETERMINISTIC]]
               [--reload_dataloaders_every_epoch [RELOAD_DATALOADERS_EVERY_EPOCH]] [--auto_lr_find [AUTO_LR_FIND]]
               [--replace_sampler_ddp [REPLACE_SAMPLER_DDP]] [--terminate_on_nan [TERMINATE_ON_NAN]] [--auto_scale_batch_size [AUTO_SCALE_BATCH_SIZE]]
               [--prepare_data_per_node [PREPARE_DATA_PER_NODE]] [--plugins PLUGINS] [--amp_backend AMP_BACKEND] [--amp_level AMP_LEVEL]
               [--distributed_backend DISTRIBUTED_BACKEND] [--automatic_optimization [AUTOMATIC_OPTIMIZATION]] [--move_metrics_to_cpu [MOVE_METRICS_TO_CPU]]
               [--enable_pl_optimizer [ENABLE_PL_OPTIMIZER]] [--feature_ext {resnet18,resnet34,resnet50,resnet101,resnet152}]
               [--use_bottleneck USE_BOTTLENECK] [--bottleneck_dim BOTTLENECK_DIM] [--new_classifier NEW_CLASSIFIER] [--random_proj RANDOM_PROJ]
               [--random_proj_dim RANDOM_PROJ_DIM] [--lr LR] [--momentum MOMENTUM] [--left_weight LEFT_WEIGHT] [--right_weight RIGHT_WEIGHT]
               [--cls_weight CLS_WEIGHT] [--mdd_weight MDD_WEIGHT] [--entropic_weight ENTROPIC_WEIGHT] [--loss_trade_off LOSS_TRADE_OFF]
               [--scheduler_lr SCHEDULER_LR] [--scheduler_weight-decay SCHEDULER_WEIGHT_DECAY] [--scheduler_gamma SCHEDULER_GAMMA]
               [--scheduler_power SCHEDULER_POWER] [--dset {office,image-clef,visda,office-home}] [--s_dset_path S_DSET_PATH] [--t_dset_path T_DSET_PATH]
               [--train_batch_size TRAIN_BATCH_SIZE] [--test_batch_size TEST_BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --logger [LOGGER]     Logger (or iterable collection of loggers) for experiment tracking.
  --checkpoint_callback [CHECKPOINT_CALLBACK]
                        If ``True``, enable checkpointing. It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint
                        in :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`. Default: ``True``. .. warning:: Passing a ModelCheckpoint
                        instance to this argument is deprecated since v1.1 and will be unsupported from v1.3. Use `callbacks` argument instead.
  --default_root_dir DEFAULT_ROOT_DIR
                        Default path for logs and weights when no logger/ckpt_callback passed. Default: ``os.getcwd()``. Can be remote file paths such as
                        `s3://mybucket/path` or 'hdfs://path/'
  --gradient_clip_val GRADIENT_CLIP_VAL
                        0 means don't clip.
  --process_position PROCESS_POSITION
                        orders the progress bar when running multiple models on same machine.
  --num_nodes NUM_NODES
                        number of GPU nodes for distributed training.
  --num_processes NUM_PROCESSES
                        number of processes for distributed training with distributed_backend="ddp_cpu"
  --gpus GPUS           number of gpus to train on (int) or which GPUs to train on (list or str) applied per node
  --auto_select_gpus [AUTO_SELECT_GPUS]
                        If enabled and `gpus` is an integer, pick available gpus automatically. This is especially useful when GPUs are configured to be in
                        "exclusive mode", such that only one process at a time can access them.
  --tpu_cores TPU_CORES
                        How many TPU cores to train on (1 or 8) / Single TPU to train on [1]
  --log_gpu_memory LOG_GPU_MEMORY
                        None, 'min_max', 'all'. Might slow performance
  --progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE
                        How often to refresh progress bar (in steps). Value ``0`` disables progress bar. Ignored when a custom callback is passed to
                        :paramref:`~Trainer.callbacks`.
  --overfit_batches OVERFIT_BATCHES
                        Overfit a percent of training data (float) or a set number of batches (int). Default: 0.0
  --track_grad_norm TRACK_GRAD_NORM
                        -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm.
  --check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                        Check val every n train epochs.
  --fast_dev_run [FAST_DEV_RUN]
                        runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es) of train, val and test to find any bugs (ie: a sort of unit test).
  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        Accumulates grads every k batches or as set up in the dict.
  --max_epochs MAX_EPOCHS
                        Stop training once this number of epochs is reached.
  --min_epochs MIN_EPOCHS
                        Force training for at least these many epochs
  --max_steps MAX_STEPS
                        Stop training after this number of steps. Disabled by default (None).
  --min_steps MIN_STEPS
                        Force training for at least these number of steps. Disabled by default (None).
  --limit_train_batches LIMIT_TRAIN_BATCHES
                        How much of training dataset to check (floats = percent, int = num_batches)
  --limit_val_batches LIMIT_VAL_BATCHES
                        How much of validation dataset to check (floats = percent, int = num_batches)
  --limit_test_batches LIMIT_TEST_BATCHES
                        How much of test dataset to check (floats = percent, int = num_batches)
  --val_check_interval VAL_CHECK_INTERVAL
                        How often to check the validation set. Use float to check within a training epoch, use int to check every n steps (batches).
  --flush_logs_every_n_steps FLUSH_LOGS_EVERY_N_STEPS
                        How often to flush logs to disk (defaults to every 100 steps).
  --log_every_n_steps LOG_EVERY_N_STEPS
                        How often to log within steps (defaults to every 50 steps).
  --accelerator ACCELERATOR
                        Previously known as distributed_backend (dp, ddp, ddp2, etc...). Can also take in an accelerator object for custom hardware.
  --sync_batchnorm [SYNC_BATCHNORM]
                        Synchronize batch norm layers between process groups/whole world.
  --precision PRECISION
                        Full precision (32), half precision (16). Can be used on CPU, GPU or TPUs.
  --weights_summary WEIGHTS_SUMMARY
                        Prints a summary of the weights when training begins.
  --weights_save_path WEIGHTS_SAVE_PATH
                        Where to save weights if specified. Will override default_root_dir for checkpoints only. Use this if for whatever reason you need the
                        checkpoints stored in a different place than the logs written in `default_root_dir`. Can be remote file paths such as
                        `s3://mybucket/path` or 'hdfs://path/' Defaults to `default_root_dir`.
  --num_sanity_val_steps NUM_SANITY_VAL_STEPS
                        Sanity check runs n validation batches before starting the training routine. Set it to `-1` to run all batches in all validation
                        dataloaders. Default: 2
  --truncated_bptt_steps TRUNCATED_BPTT_STEPS
                        Truncated back prop breaks performs backprop every k steps of much longer sequence.
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        Path/URL of the checkpoint from which training is resumed. If there is no checkpoint file at the path, start from scratch. If
                        resuming from mid-epoch checkpoint, training will start from the beginning of the next epoch.
  --profiler [PROFILER]
                        To profile individual steps during training and assist in identifying bottlenecks. Passing bool value is deprecated in v1.1 and will
                        be removed in v1.3.
  --benchmark [BENCHMARK]
                        If true enables cudnn.benchmark.
  --deterministic [DETERMINISTIC]
                        If true enables cudnn.deterministic.
  --reload_dataloaders_every_epoch [RELOAD_DATALOADERS_EVERY_EPOCH]
                        Set to True to reload dataloaders every epoch.
  --auto_lr_find [AUTO_LR_FIND]
                        If set to True, will make trainer.tune() run a learning rate finder, trying to optimize initial learning for faster convergence.
                        trainer.tune() method will set the suggested learning rate in self.lr or self.learning_rate in the LightningModule. To use a
                        different key set a string instead of True with the key name.
  --replace_sampler_ddp [REPLACE_SAMPLER_DDP]
                        Explicitly enables or disables sampler replacement. If not specified this will toggled automatically when DDP is used. By default it
                        will add ``shuffle=True`` for train sampler and ``shuffle=False`` for val/test sampler. If you want to customize it, you can set
                        ``replace_sampler_ddp=False`` and add your own distributed sampler.
  --terminate_on_nan [TERMINATE_ON_NAN]
                        If set to True, will terminate training (by raising a `ValueError`) at the end of each training batch, if any of the parameters or
                        the loss are NaN or +/-inf.
  --auto_scale_batch_size [AUTO_SCALE_BATCH_SIZE]
                        If set to True, will `initially` run a batch size finder trying to find the largest batch size that fits into memory. The result will
                        be stored in self.batch_size in the LightningModule. Additionally, can be set to either `power` that estimates the batch size through
                        a power search or `binsearch` that estimates the batch size through a binary search.
  --prepare_data_per_node [PREPARE_DATA_PER_NODE]
                        If True, each LOCAL_RANK=0 will call prepare data. Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data
  --plugins PLUGINS     Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.
  --amp_backend AMP_BACKEND
                        The mixed precision backend to use ("native" or "apex")
  --amp_level AMP_LEVEL
                        The optimization level to use (O1, O2, etc...).
  --distributed_backend DISTRIBUTED_BACKEND
                        deprecated. Please use 'accelerator'
  --automatic_optimization [AUTOMATIC_OPTIMIZATION]
                        If False you are responsible for calling .backward, .step, zero_grad in LightningModule. This argument has been moved to
                        LightningModule. It is deprecated here in v1.1 and will be removed in v1.3.
  --move_metrics_to_cpu [MOVE_METRICS_TO_CPU]
                        Whether to force internal logged metrics to be moved to cpu. This can save some gpu memory, but can make training slower. Use with
                        attention.
  --enable_pl_optimizer [ENABLE_PL_OPTIMIZER]
                        If True, each optimizer will be wrapped by `pytorch_lightning.core.optimizer.LightningOptimizer`. It allows Lightning to handle AMP,
                        TPU, accumulated_gradients, etc..

model arguments:
  --feature_ext {resnet18,resnet34,resnet50,resnet101,resnet152}
                        feature extractor type
  --use_bottleneck USE_BOTTLENECK
                        whether to use bottleneck in the classifier
  --bottleneck_dim BOTTLENECK_DIM
                        whether to use bottleneck in the classifier
  --new_classifier NEW_CLASSIFIER
                        whether to train a new classifier
  --random_proj RANDOM_PROJ
                        whether use random projection
  --random_proj_dim RANDOM_PROJ_DIM
                        random projection dimension
  --lr LR               learning rate
  --momentum MOMENTUM   momentum for the optimizer
  --left_weight LEFT_WEIGHT
  --right_weight RIGHT_WEIGHT
  --cls_weight CLS_WEIGHT
  --mdd_weight MDD_WEIGHT
  --entropic_weight ENTROPIC_WEIGHT
  --loss_trade_off LOSS_TRADE_OFF
  --scheduler_lr SCHEDULER_LR
                        learning rate for pretrained layers
  --scheduler_weight-decay SCHEDULER_WEIGHT_DECAY
                        weight decay for pretrained layers
  --scheduler_gamma SCHEDULER_GAMMA
                        gamma parameter for the inverse learning rate scheduler
  --scheduler_power SCHEDULER_POWER
                        power parameter for the inverse learning rate scheduler

data arguments:
  --dset {office-31,image-clef}
                        The dataset or source dataset type
  --s_dset_path S_DSET_PATH
                        The source dataset path list
  --t_dset_path T_DSET_PATH
                        The target dataset path list
  --train_batch_size TRAIN_BATCH_SIZE
                        Training batch size
  --test_batch_size TEST_BATCH_SIZE
                        Testing batch size
```  
where the `optional arguments` are the ones of the [Trainer](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-flags) class of [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/).  
By now, only the `office-31` and `image-clef` datasets are available, and one can train a model for example with:

* `python main.py --gpus 1 --feature_ext resnet50 --dset office-31 --s_dset_path ./data/office-31/amazon_list.txt --t_dset_path ./data/office-31/webcam_list.txt --max_steps 10000 --max_epochs 1 --default_root_dir ./chkpts --mdd_weight 0.01 --lr 0.01 --train_batch_size 32` 
* `python main.py --gpus "2,3" --feature_ext resnet50 --dset image-clef --s_dset_path ./data/image-clef/c.txt --t_dset_path ./data/image-clef/p.txt --max_steps 40000 --max_epochs 1 --default_root_dir ./chkpts --mdd_weight 0.01 --lr 0.01 --train_batch_size 32`

During training one can inspect the model behaviour with, for example, [Tensorboard](https://pytorch-lightning.readthedocs.io/en/latest/logging.html) with the following command:

* `tensorboard --logdir ./chkpts/`

# Citations

Cite the paper as follows (copied-pasted it from arxiv for you):  

```
@article{Li_2020,
   title={Maximum Density Divergence for Domain Adaptation},
   ISSN={1939-3539},
   url={http://dx.doi.org/10.1109/TPAMI.2020.2991050},
   DOI={10.1109/tpami.2020.2991050},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Li, Jingjing and Chen, Erpeng and Ding, Zhengming and Zhu, Lei and Lu, Ke and Shen, Heng Tao},
   year={2020},
   pages={1–1}
}
```

# License

This project is licensed under the MIT License

Copyright (c) 2021 Federico Belotti, Orobix Srl (www.orobix.com).

