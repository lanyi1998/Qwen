from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "./output_qwen", # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()