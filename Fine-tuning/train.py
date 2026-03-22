import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_NAME        = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # changed from base to instruct
DATASET_PATH      = "data/levantine_train.jsonl"
OUTPUT_DIR        = "outputs/qwen2.5-7b-instruct-jordanian-lora"
MAX_SEQ_LENGTH    = 512
LORA_RANK         = 32       # increased from 16
LORA_ALPHA        = 64       # higher than rank for stronger dialect push
BATCH_SIZE        = 4
GRAD_ACCUM_STEPS  = 4
LEARNING_RATE     = 2e-4
NUM_EPOCHS        = 1
SEED              = 42

# ── Load Model and Tokenizer ────────────────────────────────────────────────────

print("Loading model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name        = MODEL_NAME,
    max_seq_length    = MAX_SEQ_LENGTH,
    dtype             = None,
    load_in_4bit      = True,
)

# ── Apply LoRA ──────────────────────────────────────────────────────────────────

model = FastLanguageModel.get_peft_model(
    model,
    r                   = LORA_RANK,
    lora_alpha          = LORA_ALPHA,
    lora_dropout        = 0,
    target_modules      = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    bias                = "none",
    use_gradient_checkpointing = "unsloth",
    random_state        = SEED,
    use_rslora          = False,
    loftq_config        = None,
)

# ── Load Dataset ────────────────────────────────────────────────────────────────

print("Loading dataset...")

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

print(f"Dataset size: {len(dataset)} examples")

# ── Format Function ─────────────────────────────────────────────────────────────

QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

tokenizer.chat_template = QWEN_CHAT_TEMPLATE

def format_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_example, remove_columns=dataset.column_names)

print("Sample formatted text:")
print(dataset[0]["text"])
print("─" * 60)

# ── Training Arguments ──────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = NUM_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM_STEPS,
    learning_rate               = LEARNING_RATE,
    lr_scheduler_type           = "cosine",
    warmup_ratio                = 0.05,
    fp16                        = not torch.cuda.is_bf16_supported(),
    bf16                        = torch.cuda.is_bf16_supported(),
    logging_steps               = 10,
    save_strategy               = "epoch",
    save_total_limit            = 1,
    seed                        = SEED,
    report_to                   = "none",
    dataloader_num_workers      = 0,
)

# ── Trainer ─────────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = dataset,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LENGTH,
    args               = training_args,
)

# ── Train ────────────────────────────────────────────────────────────────────────

print("Starting training...")
trainer_stats = trainer.train()

print("\nTraining complete.")
print(f"Total steps:        {trainer_stats.global_step}")
print(f"Training loss:      {trainer_stats.training_loss:.4f}")
print(f"Runtime (seconds):  {trainer_stats.metrics['train_runtime']:.1f}")

# ── Save LoRA Adapter ────────────────────────────────────────────────────────────

print(f"\nSaving LoRA adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Adapter saved.")