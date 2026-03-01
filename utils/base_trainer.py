import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Union

from omegaconf import OmegaConf

import ignite
import ignite.distributed as idist
import torch
from ignite.contrib.engines import common
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.base_logger import BaseHandler
from ignite.engine import Engine, Events, EventEnum
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.utils import manual_seed, setup_logger
from torch.cuda.amp import autocast, GradScaler

from utils.array_operations import to
from utils.metrics import MeanMetric


def save_training_info(output_path, config):
    """
    Save training command and configuration to output path.
    
    Saves:
    - train_command.txt: The exact command used to start training
    - config.yaml: Full configuration in YAML format
    - config.json: Full configuration in JSON format
    """
    output_path = Path(output_path)
    
    # Save training command
    command = " ".join(sys.argv)
    command_file = output_path / "train_command.txt"
    with open(command_file, "w") as f:
        f.write(f"# Training command\n")
        f.write(f"# Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Working directory: {Path.cwd()}\n\n")
        f.write(f"{command}\n")
    
    # Save configuration as YAML
    config_yaml_file = output_path / "config.yaml"
    with open(config_yaml_file, "w") as f:
        f.write(f"# Training configuration\n")
        f.write(f"# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        OmegaConf.save(config, f)
    
    # Save configuration as JSON (for easy parsing)
    config_json_file = output_path / "config.json"
    config_dict = OmegaConf.to_container(config, resolve=True)
    with open(config_json_file, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    print(f"[INFO] Training info saved to {output_path}")
    print(f"  - {command_file.name}")
    print(f"  - {config_yaml_file.name}")
    print(f"  - {config_json_file.name}")

# used for debugging
torch.autograd.set_detect_anomaly(True)

def base_training(local_rank, config, get_dataflow, initialize, get_metrics, visualize):

    # copy the segmentation mode to the data and model_conf part of the config
    config['data']['segmentation_mode'] = config.get("segmentation_mode", None)
    config['model_conf']['segmentation_mode'] = config.get("segmentation_mode", None)

    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)
    device = idist.device()

    logger = setup_logger(name=config["name"])

    log_basic_info(logger, config)

    output_path = config["output_path"]
    if rank == 0:
        if config["stop_iteration"] is None:
            now = datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            now = f"stop-on-{config['stop_iteration']}"

        folder_name = f"{config['name']}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
        output_path = Path(output_path) / folder_name
        if not output_path.exists():
            output_path.mkdir(parents=True)
        config["output_path"] = output_path.as_posix()
        logger.info(f"Output path: {config['output_path']}")

        if "cuda" in device.type:
            config["cuda device name"] = torch.cuda.get_device_name(local_rank)
        
        # Save training command and configuration
        save_training_info(output_path, config)

    # Setup dataflow, model, optimizer, criterion
    loaders = get_dataflow(config, logger)
    if len(loaders) == 2:
        train_loader, test_loader = loaders
        vis_loader = None
    else:
        train_loader, test_loader, vis_loader = loaders

    if hasattr(train_loader, "dataset"):
        logger.info(f"Dataset length: Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    config["num_iters_per_epoch"] = len(train_loader)
    model, optimizer, criterion, lr_scheduler = initialize(config, logger)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Let's now setup evaluator engine to perform model's validation and compute metrics
    metrics = get_metrics(config, device)
    metrics_loss = {k: MeanMetric((lambda y: lambda x: x["loss_dict"][y])(k)) for k in criterion.get_loss_metric_names()}

    loss_during_validation = config.get("loss_during_validation", True)
    if loss_during_validation:
        eval_metrics = {**metrics, **metrics_loss}
    else:
        eval_metrics = metrics

    # Create trainer for current task
    trainer = create_trainer(model, optimizer, criterion, lr_scheduler, train_loader.sampler if hasattr(train_loader, "sampler") else None, config, logger, metrics={})

    # We define two evaluators as they wont have exactly similar roles:
    # - `evaluator` will save the best model based on validation score
    evaluator = create_evaluator(model, metrics=eval_metrics, criterion=criterion if loss_during_validation else None, config=config)

    if vis_loader is not None:
        visualizer = create_evaluator(model, metrics=eval_metrics, criterion=criterion if loss_during_validation else None, config=config)
    else:
        visualizer = None

    def run_validation(engine):
        epoch = trainer.state.epoch
        state = evaluator.run(test_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Test", state.metrics)

    def run_visualization(engine):
        epoch = trainer.state.epoch
        state = visualizer.run(vis_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Vis", state.metrics)

    eval_use_iters = config.get("eval_use_iters", False)
    vis_use_iters = config.get("vis_use_iters", False)

    if not eval_use_iters:
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=config["validate_every"]) | Events.COMPLETED, run_validation)
    else:
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=config["validate_every"]) | Events.COMPLETED, run_validation)

    if visualizer:
        if not vis_use_iters:
            trainer.add_event_handler(Events.EPOCH_COMPLETED(every=config["visualize_every"]) | Events.COMPLETED, run_visualization)
        else:
            trainer.add_event_handler(Events.ITERATION_COMPLETED(every=config["visualize_every"]) | Events.COMPLETED, run_visualization)

    if rank == 0:
        # Setup TensorBoard logging on trainer and evaluators. Logged values are:
        #  - Training metrics, e.g. running average loss values
        #  - Learning rate
        #  - Evaluation train/test metrics

        trainer_timer = IterationTimeHandler()
        trainer_timer_data = DataloaderTimeHandler()
        trainer.add_event_handler(Events.ITERATION_STARTED, trainer_timer.start_iteration)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, trainer_timer.end_iteration)
        trainer.add_event_handler(Events.GET_BATCH_STARTED, trainer_timer_data.start_get_batch)
        trainer.add_event_handler(Events.GET_BATCH_COMPLETED, trainer_timer_data.end_get_batch)

        evaluator_timer = IterationTimeHandler()
        evaluator_timer_data = DataloaderTimeHandler()
        evaluator.add_event_handler(Events.ITERATION_STARTED, evaluator_timer.start_iteration)
        evaluator.add_event_handler(Events.ITERATION_COMPLETED, evaluator_timer.end_iteration)
        evaluator.add_event_handler(Events.GET_BATCH_STARTED, evaluator_timer_data.start_get_batch)
        evaluator.add_event_handler(Events.GET_BATCH_COMPLETED, evaluator_timer_data.end_get_batch)

        if visualizer:
            visualizer_timer = IterationTimeHandler()
            visualizer_timer_data = DataloaderTimeHandler()
            visualizer.add_event_handler(Events.ITERATION_STARTED, visualizer_timer.start_iteration)
            visualizer.add_event_handler(Events.ITERATION_COMPLETED, visualizer_timer.end_iteration)
            visualizer.add_event_handler(Events.GET_BATCH_STARTED, visualizer_timer_data.start_get_batch)
            visualizer.add_event_handler(Events.GET_BATCH_COMPLETED, visualizer_timer_data.end_get_batch)

        gst = lambda engine, event_name: trainer.state.epoch
        gst_it_epoch = lambda engine, event_name: (trainer.state.epoch - 1) * engine.state.epoch_length + engine.state.iteration - 1
        eval_gst_it_iters = lambda engine, event_name: (((trainer.state.epoch - 1) * trainer.state.epoch_length + trainer.state.iteration) // config["validate_every"]) * engine.state.epoch_length + engine.state.iteration - 1
        vis_gst_it_iters =  lambda engine, event_name: (((trainer.state.epoch - 1) * trainer.state.epoch_length + trainer.state.iteration) // config["visualize_every"]) * engine.state.epoch_length + engine.state.iteration - 1

        eval_gst_ep_iters = lambda engine, event_name: (((trainer.state.epoch - 1) * trainer.state.epoch_length + trainer.state.iteration) // config["validate_every"])
        vis_gst_ep_iters = lambda engine, event_name: (((trainer.state.epoch - 1) * trainer.state.epoch_length + trainer.state.iteration) // config["visualize_every"])

        eval_gst_it = eval_gst_it_iters if eval_use_iters else gst_it_epoch
        vis_gst_it = vis_gst_it_iters if vis_use_iters else gst_it_epoch

        eval_gst_ep = eval_gst_ep_iters if eval_use_iters else gst
        vis_gst_ep = vis_gst_ep_iters if vis_use_iters else gst

        tb_logger = TensorboardLogger(log_dir=output_path)
        tb_logger.attach(trainer, MetricLoggingHandler("train", optimizer), Events.ITERATION_COMPLETED(every=config.get("log_every_iters", 1)))
        tb_logger.attach(evaluator, MetricLoggingHandler("val", log_loss=False, global_step_transform=eval_gst_ep), Events.EPOCH_COMPLETED)
        if visualizer:
            tb_logger.attach(visualizer, MetricLoggingHandler("vis", log_loss=False, global_step_transform=vis_gst_ep), Events.EPOCH_COMPLETED)

        # Plot config to tensorboard
        config_json = json.dumps(OmegaConf.to_container(config, resolve=True), indent=2)
        config_json = "".join("\t" + line for line in config_json.splitlines(True))
        tb_logger.writer.add_text("config", text_string=config_json, global_step=0)

        if visualize is not None:
            train_log_interval = config.get("log_tb_train_every_iters", -1)
            val_log_interval = config.get("log_tb_val_every_iters", train_log_interval)
            vis_log_interval = config.get("log_tb_vis_every_iters", 1)

            if train_log_interval > 0:
                tb_logger.attach(
                    trainer,
                    VisualizationHandler(tag="training", visualizer=visualize),
                    Events.ITERATION_COMPLETED(every=train_log_interval))
            if val_log_interval > 0:
                tb_logger.attach(
                    evaluator,
                    VisualizationHandler(tag="val", visualizer=visualize, global_step_transform=eval_gst_it),
                    Events.ITERATION_COMPLETED(every=val_log_interval))
            if visualizer and vis_log_interval > 0:
                tb_logger.attach(
                    visualizer,
                    VisualizationHandler(tag="vis", visualizer=visualize, global_step_transform=vis_gst_it),
                    Events.ITERATION_COMPLETED(every=vis_log_interval))

    if "save_best" in config:
        save_best_config = config["save_best"]
        
        # Support both single metric and multiple metrics
        # Single metric: {"metric": "a1", "sign": 1}
        # Multiple metrics: {"metrics": [{"metric": "a1", "sign": 1}, {"metric": "abs_rel", "sign": -1}]}
        if "metrics" in save_best_config:
            # Multiple metrics mode
            metrics_list = save_best_config["metrics"]
        else:
            # Single metric (backwards compatibility)
            metrics_list = [save_best_config]
        
        # Custom global_step_transform that directly accesses trainer's iteration
        def get_trainer_step(engine, event_name):
            return trainer.state.iteration
        
        for idx, metric_config in enumerate(metrics_list):
            metric_name = metric_config["metric"]
            sign = metric_config.get("sign", 1.0)
            
            best_model_handler = Checkpoint(
                {"model": model},
                get_save_handler(config),
                filename_prefix=f"best_{idx + 1}",
                n_saved=1,
                global_step_transform=get_trainer_step,
                score_name=metric_name,
                score_function=Checkpoint.get_default_score_fn(metric_name, score_sign=sign),
            )
            evaluator.add_event_handler(
                Events.COMPLETED, best_model_handler
            )

    # In order to check training resuming we can stop training on a given iteration
    if config["stop_iteration"] is not None:

        @trainer.on(Events.ITERATION_STARTED(once=config["stop_iteration"]))
        def _():
            logger.info(f"Stop training on {trainer.state.iteration} iteration")
            trainer.terminate()

    try:
        trainer.run(train_loader, max_epochs=config["num_epochs"])
    except Exception as e:
        logger.exception("")
        raise e

    if rank == 0:
        tb_logger.close()


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}")


def log_basic_info(logger, config):
    logger.info(f"Run {config['name']}")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as
        # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")


def create_trainer(model, optimizer, criterion, lr_scheduler, train_sampler, config, logger, metrics={}):

    device = idist.device()

    # Setup Ignite trainer:
    # - let's define training step
    # - add other common handlers:
    #    - TerminateOnNan,
    #    - handler to setup learning rate scheduling,
    #    - ModelCheckpoint
    #    - RunningAverage` on `train_step` output
    #    - Two progress bars on epochs and optionally on iterations

    with_amp = config["with_amp"]
    scaler = GradScaler(enabled=with_amp)

    def train_step(engine, data: dict):
        if "t__get_item__" in data:
            timing = {"t__get_item__": torch.mean(data["t__get_item__"]).item()}
        else:
            timing = {}

        _start_time = time.time()

        data = to(data, device)

        timing["t_to_gpu"] = time.time() - _start_time

        model.train()

        _start_time = time.time()

        with autocast(enabled=with_amp):
            data = model(data)

        timing["t_forward"] = time.time() - _start_time

        _start_time = time.time()
        loss, loss_metrics = criterion(data)
        timing["t_loss"] = time.time() - _start_time

        _start_time = time.time()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        timing["t_backward"] = time.time() - _start_time

        return {
            "output": data,
            "loss_dict": loss_metrics,
            "timings_dict": timing,
            "metrics_dict": {}
        }

    trainer = Engine(train_step)
    trainer.logger = logger

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    to_save = {"trainer": trainer, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=config["checkpoint_every"],
        save_handler=get_save_handler(config),
        lr_scheduler=lr_scheduler,
        output_names=None,
        with_pbars=False,
        clear_cuda_cache=False,
        log_every_iters=config.get("log_every_iters", 100)
    )

    resume_from = config["resume_from"]


    if resume_from is not None:
        checkpoint_fp = Path(resume_from)
        assert checkpoint_fp.exists(), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
        logger.info(f"Resume from a checkpoint: {checkpoint_fp.as_posix()}")
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    # train from backbone
    if config["name"] == "kitti_360" and config.get("use_backbone", False):
        assert not resume_from, \
            f"You are trying to load a model from '{checkpoint_fp.as_posix()}', whilst also using the backbone." \
            f" Only one of the two is possible at the same time!"
        cp_path = Path(f"out/kitti_360/pretrained")
        cp_path = next(cp_path.glob("training*.pt"))

        # TODO: Add the device back in
        cp = torch.load(cp_path)

        # Load backbone weights in a robust way and log what actually got loaded.
        # - strict=False allows missing/unexpected keys (architecture differences)
        # - shape mismatches can still raise; in that case we filter by key+shape
        if idist.get_rank() == 0:
            logger.info(f"[use_backbone] Loading pretrained backbone from: {cp_path.as_posix()}")

        state = cp["model"]

        # Handle DataParallel prefix mismatch: if model keys start with
        # "module." but checkpoint keys don't (or vice versa), remap.
        model_keys_set = set(model.state_dict().keys())
        ckpt_keys_set = set(state.keys())
        model_has_module = any(k.startswith("module.") for k in model_keys_set)
        ckpt_has_module = any(k.startswith("module.") for k in ckpt_keys_set)

        if model_has_module and not ckpt_has_module:
            state = {"module." + k: v for k, v in state.items()}
            if idist.get_rank() == 0:
                logger.info("[use_backbone] Added 'module.' prefix to checkpoint keys for DataParallel compatibility")
        elif ckpt_has_module and not model_has_module:
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            if idist.get_rank() == 0:
                logger.info("[use_backbone] Stripped 'module.' prefix from checkpoint keys")

        mismatched_keys = []
        try:
            incompatible = model.load_state_dict(state, strict=False)
        except RuntimeError as e:
            model_state = model.state_dict()
            filtered = {}
            for k, v in state.items():
                if k not in model_state:
                    continue
                if hasattr(v, "shape") and hasattr(model_state[k], "shape") and v.shape != model_state[k].shape:
                    mismatched_keys.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                    continue
                filtered[k] = v

            if idist.get_rank() == 0:
                logger.info(
                    f"[use_backbone] Initial load_state_dict failed (likely shape mismatch). "
                    f"Retrying with filtered state_dict (kept {len(filtered)}/{len(state)} tensors). Error: {e}"
                )
            incompatible = model.load_state_dict(filtered, strict=False)

        if idist.get_rank() == 0:
            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            model_keys = list(model.state_dict().keys())
            ckpt_keys = list(state.keys())
            loaded_keys = [k for k in model_keys if (k in state)]

            logger.info(
                f"[use_backbone] Loaded. model_keys={len(model_keys)}, ckpt_keys={len(ckpt_keys)}, "
                f"loaded_keys={len(loaded_keys)}, missing_keys={len(missing)}, unexpected_keys={len(unexpected)}, "
                f"shape_mismatched_filtered={len(mismatched_keys)}"
            )

            # Print a small sample for quick inspection (full lists can be huge)
            if loaded_keys:
                logger.info("[use_backbone] Sample loaded_keys (up to 30):\n\t" + "\n\t".join(loaded_keys[:30]))
            if missing:
                logger.info("[use_backbone] Sample missing_keys (up to 30):\n\t" + "\n\t".join(missing[:30]))
            if unexpected:
                logger.info("[use_backbone] Sample unexpected_keys (up to 30):\n\t" + "\n\t".join(unexpected[:30]))
            if mismatched_keys:
                pretty = [f"{k}: ckpt{cs} != model{ms}" for k, cs, ms in mismatched_keys[:30]]
                logger.info("[use_backbone] Sample shape mismatches filtered (up to 30):\n\t" + "\n\t".join(pretty))

    return trainer


def create_evaluator(model, metrics, criterion, config, tag="val"):
    with_amp = config["with_amp"]
    device = idist.device()

    @torch.no_grad()
    def evaluate_step(engine: Engine, data):
        model.eval()
        if "t__get_item__" in data:
            timing = {"t__get_item__": torch.mean(data["t__get_item__"]).item()}
        else:
            timing = {}

        data = to(data, device)

        with autocast(enabled=with_amp):
            data = model(data)

        for name in metrics.keys():
            data[name] = data[name].mean()

        if criterion is not None:
            loss, loss_metrics = criterion(data)
        else:
            loss_metrics = {}

        return {
            "output": data,
            "loss_dict": loss_metrics,
            "timings_dict": timing,
            "metrics_dict": {}
        }

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    if idist.get_rank() == 0 and (not config.get("with_clearml", False)):
        common.ProgressBar(desc=f"Evaluation ({tag})", persist=False).attach(evaluator)

    return evaluator


def get_save_handler(config):
    return DiskSaver(config["output_path"], require_empty=False)


class MetricLoggingHandler(BaseHandler):
    def __init__(self, tag, optimizer=None, log_loss=True, log_metrics=True, log_timings=True, global_step_transform=None):
        self.tag = tag
        self.optimizer = optimizer
        self.log_loss = log_loss
        self.log_metrics = log_metrics
        self.log_timings = log_timings
        self.gst = global_step_transform
        super(MetricLoggingHandler, self).__init__()

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, EventEnum]):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'MetricLoggingHandler' works only with TensorboardLogger")

        if self.gst is None:
            gst = global_step_from_engine(engine)
        else:
            gst = self.gst
        global_step = gst(engine, event_name)  # type: ignore[misc]

        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        writer = logger.writer

        # Optimizer parameters
        if self.optimizer is not None:
            params = {
                k: float(param_group["lr"]) for k, param_group in enumerate(self.optimizer.param_groups)
            }

            for k, param in params.items():
                writer.add_scalar(f"lr-{self.tag}/{k}", param, global_step)

        if self.log_loss:
            # Plot losses
            loss_dict = engine.state.output["loss_dict"]
            for k, v in loss_dict.items():
                writer.add_scalar(f"loss-{self.tag}/{k}", v, global_step)

        if self.log_metrics:
            # Plot metrics
            metrics_dict = engine.state.metrics
            metrics_dict_custom = engine.state.output["metrics_dict"]

            for k, v in metrics_dict.items():
                writer.add_scalar(f"metrics-{self.tag}/{k}", v, global_step)
            for k, v in metrics_dict_custom.items():
                writer.add_scalar(f"metrics-{self.tag}/{k}", v, global_step)

        if self.log_timings:
            # Plot timings
            timings_dict = engine.state.times
            timings_dict_custom = engine.state.output["timings_dict"]
            for k, v in timings_dict.items():
                if k == "COMPLETED":
                    continue
                writer.add_scalar(f"timing-{self.tag}/{k}", v, global_step)
            for k, v in timings_dict_custom.items():
                writer.add_scalar(f"timing-{self.tag}/{k}", v, global_step)


class IterationTimeHandler:
    def __init__(self):
        self._start_time = None

    def start_iteration(self, engine):
        self._start_time = time.time()

    def end_iteration(self, engine):
        if self._start_time is None:
            t_diff = 0
            iters_per_sec = 0
        else:
            t_diff = max(time.time() - self._start_time, 1e-6)
            iters_per_sec = 1 / t_diff
        if not hasattr(engine.state, "times"):
            engine.state.times = {}
        else:
            engine.state.times["secs_per_iter"] = t_diff
            engine.state.times["iters_per_sec"] = iters_per_sec


class DataloaderTimeHandler:
    def __init__(self):
        self._start_time = None

    def start_get_batch(self, engine):
        self._start_time = time.time()

    def end_get_batch(self, engine):
        if self._start_time is None:
            t_diff = 0
            iters_per_sec = 0
        else:
            t_diff = max(time.time() - self._start_time, 1e-6)
            iters_per_sec = 1 / t_diff
        if not hasattr(engine.state, "times"):
            engine.state.times = {}
        else:
            engine.state.times["get_batch_secs"] = t_diff


class VisualizationHandler(BaseHandler):
    def __init__(self, tag, visualizer, global_step_transform=None):
        self.tag = tag
        self.visualizer = visualizer
        self.gst = global_step_transform
        super(VisualizationHandler, self).__init__()

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, EventEnum]) -> None:

        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'VisualizationHandler' works only with TensorboardLogger")

        if self.gst is None:
            gst = global_step_from_engine(engine)
        else:
            gst = self.gst
        global_step = gst(engine, event_name)  # type: ignore[misc]

        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        self.visualizer(engine, logger, global_step, self.tag)
