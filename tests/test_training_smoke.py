"""Optional CPU-only wiring check; this is not a d1-3B training result."""

import importlib.util
import os
import shutil
import sys

import pytest

pytestmark = pytest.mark.model


@pytest.mark.skipif(
    os.environ.get("TOUCH_GRASS_SMOKE") != "1",
    reason="Set TOUCH_GRASS_SMOKE=1 with model/train dependencies",
)
def test_answer_slot_training_merge_and_native_reload(tmp_path):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from touch_grass.dataset import BASE_MODEL, load_rows, prompt_completion, question
    from touch_grass.judge import Judge, checkpoint_path

    torch.set_num_threads(2)
    local = checkpoint_path(BASE_MODEL)
    spec = importlib.util.spec_from_file_location("_d1_smoke_prompt", local / "prompt.py")
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    tokenizer = AutoTokenizer.from_pretrained(local)
    rows = load_rows("data/train.jsonl")[:2]
    pairs = [prompt_completion(row, tokenizer, native) for row in rows]
    config = AutoConfig.from_pretrained(local)
    config.text_config.hidden_size = 32
    config.text_config.intermediate_size = 64
    config.text_config.num_hidden_layers = 2
    config.text_config.layer_types = ["conv", "full_attention"]
    config.text_config.num_attention_heads = 4
    config.text_config.num_key_value_heads = 2
    config.text_config.block_multiple_of = 8
    config.vision_config.hidden_size = 32
    config.vision_config.intermediate_size = 64
    config.vision_config.num_hidden_layers = 1
    config.vision_config.num_attention_heads = 4
    config.projector_hidden_size = 32
    config.use_cache = False
    model = AutoModelForImageTextToText.from_config(config)
    model = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
            exclude_modules=r".*vision_tower.*",
        ),
    )
    assert not any(
        p.requires_grad for name, p in model.named_parameters() if "vision_tower" in name
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(pairs),
        args=SFTConfig(
            output_dir=str(tmp_path / "adapter"),
            use_cpu=True,
            bf16=False,
            fp16=False,
            per_device_train_batch_size=1,
            max_steps=1,
            save_strategy="no",
            max_length=2048,
            completion_only_loss=True,
            packing=False,
            report_to="none",
            optim="adamw_torch",
            dataloader_pin_memory=False,
        ),
    )
    batch = trainer.data_collator([trainer.train_dataset[0]])
    expected = tokenizer.encode(pairs[0]["completion"], add_special_tokens=False)
    assert len(expected) == 2
    assert batch["labels"][0][batch["labels"][0] != -100].tolist() == expected
    # TRL 1.15's fused training kernel requires CUDA; check CPU gradients without that kernel.
    optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
    loss = trainer.model(**batch, fused_lm_head=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert any(torch.count_nonzero(p) for name, p in model.named_parameters() if "lora_B" in name)
    merged_path = tmp_path / "merged"
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
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
        shutil.copy2(local / filename, merged_path / filename)
    judge = Judge(str(merged_path), "cpu")
    native_render = judge.model.engine.render(
        {"task": rows[0]["task"], "trajectory": rows[0]["trajectory"]},
        judge.native_prompt.as_question(question(rows[0]["candidates"])),
    )
    assert native_render == pairs[0]["prompt"]
    result = judge.predict(rows)[0]
    assert result["usage"]["output_tokens"] == 0
    assert sum(result["probabilities"].values()) == pytest.approx(1)
