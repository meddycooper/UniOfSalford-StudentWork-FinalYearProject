from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./test_output",
    evaluation_strategy="epoch"
)

print("Passed: evaluation_strategy accepted.")