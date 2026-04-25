# This file needs to be run in the main folder
# %%
import text
import os
from utils import read_lines_from_file

def write_lines_to_file(path, lines, mode='w', encoding='utf-8'):
    with open(path, mode, encoding=encoding) as f:
        for i, line in enumerate(lines):
            if i == len(lines)-1:
                f.write(line)
                break
            f.write(line + '\n')

# %%
# 1. Point directly to your new Jordanian dataset
metadata_path = 'C:/Users/20220020/Desktop/TTS GP/jordanian_tts_data/metadata.csv'
print(f"Reading from {metadata_path}...")
lines = read_lines_from_file(metadata_path)

new_lines_arabic = []
new_lines_phonetic = []
new_lines_buckw = []

for line in lines:
    if not line.strip():
        continue
        
    # 2. Split using the pipe '|' from our generated metadata
    try:
        wav_name, utterance_arab = line.strip().split('|')
    except ValueError:
        print(f"Skipping malformed line: {line}")
        continue

    # 3. Convert Standard Arabic -> Buckwalter -> Phonemes
    utterance_buckw = text.arabic_to_buckwalter(utterance_arab)
    
    # Apply the developer's original diacritic adjustments
    utterance_buckw = utterance_buckw.replace("a~", "~a") \
                                     .replace("i~", "~i") \
                                     .replace("u~", "~u") \
                                     .replace(" - ", " ")

    utterance_phon = text.buckwalter_to_phonemes(utterance_buckw)

    # 4. Format them back into the strict "filename" "phonemes" structure FastPitch demands
    line_new_ara = f'"{wav_name}" "{utterance_arab}"'
    new_lines_arabic.append(line_new_ara)

    line_new_pho = f'"{wav_name}" "{utterance_phon}"'
    new_lines_phonetic.append(line_new_pho)

    line_new_buckw = f'"{wav_name}" "{utterance_buckw}"'
    new_lines_buckw.append(line_new_buckw)

# %% Save the newly processed files
output_dir = 'C:/Users/20220020/Desktop/TTS GP/jordanian_tts_data/processed'
os.makedirs(output_dir, exist_ok=True)

write_lines_to_file(os.path.join(output_dir, 'train_arab.txt'), new_lines_arabic)
write_lines_to_file(os.path.join(output_dir, 'train_phon.txt'), new_lines_phonetic)
write_lines_to_file(os.path.join(output_dir, 'train_buckw.txt'), new_lines_buckw)

# Since we aren't splitting train/test yet, we just duplicate for testing purposes
write_lines_to_file(os.path.join(output_dir, 'test_arab.txt'), new_lines_arabic)
write_lines_to_file(os.path.join(output_dir, 'test_phon.txt'), new_lines_phonetic)
write_lines_to_file(os.path.join(output_dir, 'test_buckw.txt'), new_lines_buckw)

print(f"✅ Text preprocessing complete! Processed {len(new_lines_phonetic)} audio labels.")
print(f"Files saved to: {output_dir}")