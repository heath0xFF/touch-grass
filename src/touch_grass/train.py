import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

from touch_grass.dataset import (
    BASE_MODEL,
    BASE_REVISION,
    MAX_TOKENS,
    check_disjoint,
    load_rows,
    prompt_completion,
)
from touch_grass.judge import checkpoint_path


def train(config_path):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    from touch_grass.evaluate import environment

    config = json.loads(Path(config_path).read_text())
    output = Path(config["output_dir"])
    if output.exists():
        raise ValueError(f"Refusing to overwrite checkpoint: {output}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Training requires a BF16-capable CUDA GPU")
    set_seed(config["seed"])
    train_rows = load_rows(config["train_file"])
    validation_rows = load_rows(config["validation_file"])
    check_disjoint(
        train_rows, validation_rows, load_rows("data/test.jsonl"), load_rows("data/gold.jsonl")
    )
    local = checkpoint_path(BASE_MODEL)
    spec = importlib.util.spec_from_file_location("_d1_prompt", local / "prompt.py")
    native_prompt = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native_prompt
    spec.loader.exec_module(native_prompt)
    tokenizer = AutoTokenizer.from_pretrained(local)
    datasets = [
        Dataset.from_list([prompt_completion(row, tokenizer, native_prompt) for row in rows])
        for rows in (train_rows, validation_rows)
    ]
    # Train the native answer slot using Liquid's standard-backbone recipe.
    model = AutoModelForImageTextToText.from_pretrained(
        local, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
            exclude_modules=r".*vision_tower.*",
        ),
    )
    model.print_trainable_parameters()
    args = SFTConfig(
        output_dir=str(output / "adapter"),
        num_train_epochs=1,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=True,
        fp16=False,
        report_to="none",
        seed=config["seed"],
        max_length=MAX_TOKENS,
        completion_only_loss=True,
        packing=False,
        optim="adamw_torch",
        dataloader_num_workers=0,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=datasets[0],
        eval_dataset=datasets[1],
        processing_class=tokenizer,
    )
    # A masking regression can silently turn decision training into language modeling.
    batch = trainer.data_collator([trainer.train_dataset[0]])
    supervised = batch["labels"][0][batch["labels"][0] != -100].tolist()
    expected = tokenizer.encode(datasets[0][0]["completion"], add_special_tokens=False)
    if supervised != expected or len(supervised) != 2:
        raise ValueError(
            f"Expected option-code + EOS supervision, got {supervised}; expected {expected}"
        )
    result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output / "adapter")
    merged = trainer.model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(output / "merged")
    tokenizer.save_pretrained(output / "merged")
    for filename in (
        "api.py",
        "hybrid.py",
        "lfm2_vl.py",
        "modeling_d1.py",
        "prompt.py",
        "runner.py",
        "processor_config.json",
        "LICENSE",
    ):
        shutil.copy2(local / filename, output / "merged" / filename)
    provenance = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "config": config,
        "environment": environment(),
        "training_packages": {
            name: __import__(name).__version__ for name in ("trl", "peft", "datasets")
        },
        "data_sha256": {
            key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
            for key in ("train_file", "validation_file")
        },
        "supervised_token_ids_check": supervised,
        "metrics": result.metrics,
        "history": trainer.state.log_history,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    for directory in (output / "adapter", output / "merged"):
        shutil.copy2(local / "LICENSE", directory / "LICENSE")
        (directory / "NOTICE").write_text(
            "Derived from LiquidAI/d1-3B under the LFM Open License v1.0.\n"
            "Weights modified by touch-grass: language-only LoRA on synthetic agent trajectories.\n"
            "Configuration and tokenizer files were serialized by the training libraries.\n"
            "Native d1 Python modules are unchanged. See training.json for provenance.\n"
        )
        (directory / "training.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "training.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Native d1 checkpoint saved: {output / 'merged'}", flush=True)
