import math, os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def _tracker_requested(args) -> bool:
    if args is None:
        return False
    report_to = getattr(args, "report_to", None)
    if report_to is None:
        return False
    trackers = [item.strip().lower() for item in report_to.split(",") if item.strip()]
    trackers = [item for item in trackers if item != "none"]
    return len(trackers) > 0


def _build_lr_scheduler(optimizer, args, total_update_steps):
    scheduler_type = getattr(args, "lr_scheduler", "constant") if args is not None else "constant"
    if scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer)

    if scheduler_type != "cosine":
        raise ValueError(f"Unsupported lr_scheduler: {scheduler_type}")

    warmup_steps = max(0, int(getattr(args, "lr_warmup_steps", 0) or 0))
    min_ratio = float(getattr(args, "lr_min_ratio", 0.0) or 0.0)
    min_ratio = min(max(min_ratio, 0.0), 1.0)
    total_update_steps = max(1, int(total_update_steps))

    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        cosine_steps = max(1, total_update_steps - warmup_steps)
        progress = min(max((current_step - warmup_steps) / cosine_steps, 0.0), 1.0)
        cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _set_dataloader_epoch(dataloader, epoch_id):
    candidates = [dataloader]
    for attr_name in ("sampler", "batch_sampler"):
        candidate = getattr(dataloader, attr_name, None)
        if candidate is not None:
            candidates.append(candidate)
            nested_sampler = getattr(candidate, "sampler", None)
            if nested_sampler is not None:
                candidates.append(nested_sampler)
    for candidate in candidates:
        set_epoch = getattr(candidate, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch_id)


def _save_training_state(
    accelerator,
    output_path,
    optimizer,
    scheduler,
    global_step,
    next_epoch_id,
    next_batch_id,
    lora_checkpoint,
    args,
):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    os.makedirs(output_path, exist_ok=True)
    state = {
        "global_step": int(global_step),
        "next_epoch_id": int(next_epoch_id),
        "next_batch_id": int(next_batch_id),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "lora_checkpoint": lora_checkpoint,
        "num_epochs": int(getattr(args, "num_epochs", 0) or 0) if args is not None else None,
        "lr_scheduler": getattr(args, "lr_scheduler", None) if args is not None else None,
        "learning_rate": float(getattr(args, "learning_rate", 0.0) or 0.0) if args is not None else None,
    }
    latest_path = os.path.join(output_path, "training-state-latest.pt")
    step_path = os.path.join(output_path, f"training-state-step-{global_step}.pt")
    accelerator.save(state, latest_path)
    accelerator.save(state, step_path)
    print(f"Training state saved: {latest_path}")


def _load_training_state(accelerator, optimizer, scheduler, model_logger, args):
    resume_path = getattr(args, "resume_training_state", None) if args is not None else None
    if not resume_path:
        return 0, 0, 0
    state = torch.load(resume_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    global_step = int(state.get("global_step", 0))
    next_epoch_id = int(state.get("next_epoch_id", 0))
    next_batch_id = int(state.get("next_batch_id", 0))
    model_logger.num_steps = global_step
    if accelerator.is_main_process:
        print(
            f"Training state loaded: {resume_path} | "
            f"global_step={global_step}, next_epoch={next_epoch_id}, next_batch={next_batch_id}"
        )
    return global_step, next_epoch_id, next_batch_id


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
    dataloader: torch.utils.data.DataLoader = None,
):
    max_train_steps = None
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        max_train_steps = getattr(args, "max_train_steps", None)

    if hasattr(model, "optimizer_param_groups"):
        optimizer = torch.optim.AdamW(model.optimizer_param_groups(learning_rate, weight_decay))
    else:
        optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    if dataloader is None:
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    update_steps_per_epoch = math.ceil(len(dataloader) / accelerator.gradient_accumulation_steps)
    total_update_steps = max_train_steps if max_train_steps is not None else update_steps_per_epoch * num_epochs
    scheduler = _build_lr_scheduler(optimizer, args, total_update_steps)
    if accelerator.is_main_process:
        scheduler_type = getattr(args, "lr_scheduler", "constant") if args is not None else "constant"
        warmup_steps = getattr(args, "lr_warmup_steps", 0) if args is not None else 0
        min_ratio = getattr(args, "lr_min_ratio", 0.0) if args is not None else 0.0
        print(
            f"LR scheduler: {scheduler_type}, total_update_steps={total_update_steps}, "
            f"warmup_steps={warmup_steps}, min_ratio={min_ratio}"
        )

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    global_step, start_epoch_id, resume_batch_id = _load_training_state(
        accelerator,
        optimizer,
        scheduler,
        model_logger,
        args,
    )
    tracking_enabled = _tracker_requested(args)
    next_epoch_to_save = start_epoch_id
    next_batch_to_save = resume_batch_id
    
    for epoch_id in range(start_epoch_id, num_epochs):
        _set_dataloader_epoch(dataloader, epoch_id)
        for batch_id, data in enumerate(tqdm(dataloader, disable=not accelerator.is_local_main_process)):
            if epoch_id == start_epoch_id and batch_id < resume_batch_id:
                continue
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                # if dataset.load_from_cache:
                #     loss = model({}, inputs=data)
                # else:
                loss = model(data)
                # print(len(loss))
                # print(f"[DEBUG] loss: {loss}")                
                accelerator.backward(loss)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                    global_step += 1
                    model_logger.on_step_end(accelerator, model, save_steps)
                    next_epoch_to_save = epoch_id
                    next_batch_to_save = batch_id + 1
                    if next_batch_to_save >= len(dataloader):
                        next_epoch_to_save = epoch_id + 1
                        next_batch_to_save = 0
                    if save_steps is not None and global_step % save_steps == 0:
                        lora_checkpoint = os.path.join(model_logger.output_path, f"step-{global_step}.safetensors")
                        _save_training_state(
                            accelerator,
                            model_logger.output_path,
                            optimizer,
                            scheduler,
                            global_step,
                            next_epoch_to_save,
                            next_batch_to_save,
                            lora_checkpoint,
                            args,
                        )
                    if tracking_enabled:
                        reduced_loss = accelerator.reduce(loss.detach(), reduction="mean").item()
                        metrics = {
                            "train/loss": reduced_loss,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": float(epoch_id + 1),
                            "train/global_step": float(global_step),
                        }
                        unwrapped_model = accelerator.unwrap_model(model)
                        pipe = getattr(unwrapped_model, "pipe", None)
                        latest_loss_info = getattr(pipe, "latest_loss_info", None)
                        if isinstance(latest_loss_info, dict):
                            loss_type = latest_loss_info.get("type", "loss").lower()
                            for key, value in latest_loss_info.get("values", {}).items():
                                if isinstance(value, (int, float)):
                                    value_tensor = torch.tensor(value, device=accelerator.device, dtype=torch.float32)
                                    metrics[f"train/{loss_type}/{key}"] = accelerator.reduce(value_tensor, reduction="mean").item()
                        if accelerator.trackers:
                            accelerator.log(metrics, step=global_step)
                    if max_train_steps is not None and global_step >= max_train_steps:
                        break
        if max_train_steps is not None and global_step >= max_train_steps:
            break
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
            lora_checkpoint = os.path.join(model_logger.output_path, f"epoch-{epoch_id}.safetensors")
            _save_training_state(
                accelerator,
                model_logger.output_path,
                optimizer,
                scheduler,
                global_step,
                epoch_id + 1,
                0,
                lora_checkpoint,
                args,
            )
    model_logger.on_training_end(accelerator, model, save_steps)
    final_lora_checkpoint = os.path.join(model_logger.output_path, f"step-{model_logger.num_steps}.safetensors")
    _save_training_state(
        accelerator,
        model_logger.output_path,
        optimizer,
        scheduler,
        global_step,
        next_epoch_to_save,
        next_batch_to_save,
        final_lora_checkpoint,
        args,
    )
    accelerator.end_training()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
