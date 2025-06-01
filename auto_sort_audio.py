import os
import shutil

source_dir = r"C:\speech_recognition\dataset"
dest_dir = r"C:speech_recognition\processed_data"

emotion_map = {
    '01': 'neutral',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry'
}

for emotion in emotion_map.values():
    os.makedirs(os.path.join(dest_dir, emotion), exist_ok=True)

for actor_folder in os.listdir(source_dir):
    actor_path = os.path.join(source_dir, actor_folder)

    if os.path.isdir(actor_path):
        for filename in os.listdir(actor_path):
            if filename.endswith(".wav"):
                parts = filename.split("-")
                emotion_code = parts[2]

                if emotion_code in emotion_map:
                    emotion = emotion_map[emotion_code]
                    src_file = os.path.join(actor_path, filename)
                    dst_file = os.path.join(dest_dir, emotion, filename)

                    shutil.copyfile(src_file, dst_file)

print("All files organized into emotion folders.")
