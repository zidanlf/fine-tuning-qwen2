# Fine-Tuning Qwen2-1.5B dengan Unsloth untuk Code Generation
Proyek ini melakukan fine-tuning model Qwen2-1.5B menggunakan Unsloth library untuk meningkatkan kemampuan code generation Python. Model dilatih menggunakan dataset Alpaca dan Evol-Instruct-Python-26k.

## 🚀 Features
- Fine-tuning model Qwen2-1.5B dengan LoRA (Low-Rank Adaptation)
- Optimasi memory menggunakan 4-bit quantization
- Support untuk dataset Parquet dan Excel
- Evaluasi sebelum dan sesudah training
- Ekstraksi otomatis kode Python dari response model
- Text streaming untuk inference real-time

## 📋 Requirements
System Requirements:
- Python 3.8+
- CUDA-capable GPU (recommended)
- Minimum 8GB GPU memory

## Dependencies
Install dependencies menggunakan:
bashpip install -r requirements.txt
pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

## 🛠️ Installation

Clone repository ini:

bashgit clone <repository-url>
cd <repository-name>

Install dependencies:

bashpip install -r requirements.txt
pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

Siapkan dataset dalam format Parquet:

train-00000-of-00001.parquet
test-00000-of-00001.parquet
validation-00000-of-00001.parquet
prompt-00000-of-00001.parquet



📊 Dataset
Proyek ini menggunakan dua dataset utama:

Dataset Lokal: File Parquet yang berisi prompt dan response untuk evaluasi
Evol-Instruct-Python-26k: Dataset dari Hugging Face untuk training

26,000 contoh instruksi Python
Format: instruction -> output



Format Dataset
{
  "instruction": "Tulis fungsi Python untuk...",
  "output": "def function_name():\n    # kode di sini"
}
🏃 Run
Jalankan script Python untuk memulai fine-tuning dan evaluasi model.
📈 Evaluation
Script akan menghasilkan beberapa file evaluasi:

all_before1.xlsx: Response model sebelum fine-tuning
all_after1.xlsx: Response model setelah fine-tuning
source_code_python_valPrompt_before.xlsx: Kode Python yang diekstrak sebelum training
source_code_python_valPrompt_after.xlsx: Kode Python yang diekstrak setelah training

Ekstraksi Kode Python
pythondef extract_python_code(response):
    matches = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
    if matches:
        return matches[0].strip()
    else:
        return None
🎯 Model Configuration
LoRA Parameters

r: 16 (rank)
lora_alpha: 16
lora_dropout: 0
Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

Training Parameters

Batch size: 2
Gradient accumulation steps: 8
Learning rate: 5e-5
Max steps: 120
Warmup steps: 20
Optimizer: AdamW 8-bit

📝 Prompt Template
Model menggunakan template Alpaca untuk formatting:
write code to python based on input.

Input:
{instruction}

Response:
{output}
🔍 Memory Usage
Script menyediakan monitoring memory GPU:
pythongpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
🚨 Troubleshooting
Common Issues

CUDA Out of Memory: Kurangi per_device_train_batch_size atau max_seq_length
Slow Training: Pastikan menggunakan use_gradient_checkpointing="unsloth"
Import Error: Pastikan Unsloth terinstall dengan benar dari GitHub

Performance Tips

Gunakan fp16=True atau bf16=True untuk memory efficiency
Set packing=False untuk dataset dengan variasi panjang
Gunakan dataset_num_proc=2 untuk parallel processing

📄 License
[Sesuaikan dengan license yang Anda gunakan]
🤝 Contributing

Fork repository
Buat feature branch
Commit changes
Push ke branch
Create Pull Request

📧 Contact
[Masukkan informasi kontak Anda]
🙏 Acknowledgments

Unsloth - Fast LLM fine-tuning
Qwen2 - Base model
Evol-Instruct-Python-26k - Training dataset
