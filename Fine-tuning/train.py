from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# 1. Configuration
max_seq_length = 2048
dtype = None # Auto-detect
load_in_4bit = True # Crucial to fit Qwen 2.5 7B into your 16GB VRAM

print("Loading Model...")
# 2. Load Qwen 2.5 7B Instruct Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Apply LoRA (Trains only a tiny fraction of the model)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules =["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,  
)

# 4. Prepare Dataset Formatting
print("Preparing Dataset...")
def format_prompts(example):
    formatted_texts = []
    
    for inst, ctx, res in zip(example['instruction'], example['context'], example['response']):
        
        # Hardcode the system prompt here
        sys_prompt = """أنت مساعد ذكي لجامعة  . أجب بدقة وبشكل مفيد بناءً على السياق المقدم.
        هذا مثلال عن طريقة الحوار:
    "instruction": "يعطيك العافية كم أكثر شي بقدر أنزل ساعات بالفصل العادي؟",
    "context": "يكون العبء الدراسي للطالب في الجامعة (18) ثماني عشرة ساعة معتمدة في الفصل الدراسي حداً أقصى. ويجوز أن يأخذ الطالب في الجامعة ثلاث ساعات إضافية، إذا كان معدله التراكمي في الفصل السابق لا يقل عن 80%، بما في ذلك الفصل الصيفي.",
    "response": "الله يعافيك يا هلا. الحد الأقصى للساعات بالفصل العادي هو 18 ساعة معتمدة. بس إذا كنت شطور ومعدلك التراكمي بالفصل الماضي 80% فأكثر، بتقدر تنزل 3 ساعات زيادة يعني بصيروا 21 ساعة. بالتوفيق!"
  """
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"{inst}\n\n[السياق للمساعدة]:\n{ctx}"},
            {"role": "assistant", "content": res}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted_texts.append(text)
        
    return {"text": formatted_texts}

# Load your local JSON file
dataset = load_dataset("json", data_files="data.json", split="train")
dataset = dataset.map(format_prompts, batched=True)

# 5. Training Setup
print("Starting Training...")
trainer = SFTTrainer(
    model = model,
    r = 32,
    target_modules = "all-linear",
    lora_alpha = 32,
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2, 
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 4,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# Run the training
trainer_stats = trainer.train()

# 6. Export to Ollama GGUF format
print("Training Complete! Exporting to Ollama GGUF Format...")
# Quantize to Q4_K_M so it runs blazingly fast in Ollama
model.save_pretrained_gguf("university_bot_model", tokenizer, quantization_method = "q4_k_m")

print("Export Successful! You can now load it into Ollama.")