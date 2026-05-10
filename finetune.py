from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

max_seq_length = 2048
dtype          = None 
load_in_4bit   = True 

# 1. Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = "unsloth/Qwen2.5-7B-Instruct",
    max_seq_length  = max_seq_length,
    dtype           = dtype,
    load_in_4bit    = load_in_4bit,
)

# 2. Add LoRA Adapters (Reduced 'r' for stability on small datasets)
model = FastLanguageModel.get_peft_model(
    model,
    r                   = 16, # Lower rank is often more stable for <2k rows
    lora_alpha          = 32,
    target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
    lora_dropout        = 0.05, # Added small dropout to prevent overfitting
    bias                = "none",
    use_gradient_checkpointing = "unsloth",
    random_state        = 3407,
)

# 3. Improved System Prompt & Formatting
SYSTEM_PROMPT = """أنت مساعد ذكي لجامعة أردنية. 
مهمتك الإجابة على استفسارات الطلاب بلهجة أردنية محترمة وودودة.
استخدم كلمات مثل (بقدر، عشان، اه، تمام، ولا يهمك).
اعتمد فقط على المعلومات المزودة في [السياق]. إذا لم يكن السؤال ضمن اختصاصك أو لا يوجد سياق كافٍ، اعتذر بلطف."""

def format_prompts(example):
    texts = []
    for inst, ctx, res in zip(example["instruction"], example["context"], example["response"]):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            # Explicitly labeling the context helps the model separate facts from the dialect style
            {"role": "user", "content": f"السؤال: {inst}\n\n[المعلومات المتاحة]:\n{ctx}"},
            {"role": "assistant", "content": res},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

# 4. Dataset Loading
dataset = load_dataset("json", data_files="data.json", split="train")
dataset = dataset.map(format_prompts, batched=True)

# 5. Trainer (Lower Learning Rate is KEY)
trainer = SFTTrainer(
    model             = model,
    tokenizer         = tokenizer,
    train_dataset     = dataset,
    dataset_text_field= "text",
    max_seq_length    = max_seq_length,
    dataset_num_proc  = 2,
    args = TrainingArguments(
        per_device_train_batch_size  = 2,
        gradient_accumulation_steps  = 4,
        warmup_ratio                 = 0.1,
        num_train_epochs             = 1,      
        learning_rate                = 5e-5,   
        fp16                         = not is_bfloat16_supported(),
        bf16                         = is_bfloat16_supported(),
        logging_steps                = 1,
        optim                        = "adamw_8bit",
        weight_decay                 = 0.01,
        lr_scheduler_type            = "cosine",
        seed                         = 3407,
        output_dir                   = "outputs",
    ),
)

trainer.train()

# 6. Export directly to GGUF (Optimized for Ollama)
model.save_pretrained_gguf("jordanian_uni_model", tokenizer, quantization_method="q4_k_m")