import json

FILEPATH = "cleaned_dialect.jsonl"

errors = []
total = 0
dialect_examples = 0
qa_examples = 0

with open(FILEPATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            messages = obj.get("messages", [])

            # Check structure
            roles = [m["role"] for m in messages]
            if roles[0] != "system":
                errors.append(f"Line {i}: first message is not system")
            if messages[0]["content"] != "تحدث باللهجة الأردنية العامية دائماً":
                errors.append(f"Line {i}: wrong system message")

            # Count type
            if len(messages) == 2:
                dialect_examples += 1
            elif len(messages) == 3:
                qa_examples += 1

            total += 1
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: JSON error - {e}")

print(f"Total examples: {total:,}")
print(f"  Pre-training (system+assistant): {dialect_examples:,}")
print(f"  Q&A pairs (system+user+assistant): {qa_examples:,}")
print(f"  Errors found: {len(errors)}")

if errors:
    print("\nFirst 10 errors:")
    for e in errors[:10]:
        print(f"  {e}")
else:
    print("\nAll examples passed validation.")