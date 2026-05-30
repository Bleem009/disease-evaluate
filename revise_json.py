import json
from pathlib import Path

# 定义文件名关键词到新标签的映射
keyword_to_label = {
    'mildew': 'powdery',
    'rust': 'rust',#
    'blast': 'rust',#
    'sheath_blight':'rust',#
    'spot': 'spot',
    'bacterial_blight':'bacterial_blight',
    'brown_spot':'bacterial_blight',
    # 添加更多映射
}


def update_json_label(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 根据文件名选择新标签
    filename = json_path.stem
    new_label = None
    for keyword, label in keyword_to_label.items():
        if keyword in filename:
            new_label = label
            break
    if new_label is None:
        print(f"Warning: No keyword matched for {json_path.name}, skipping.")
        return False

    # 修改所有 shapes 的 label
    modified = False
    for shape in data.get('shapes', []):
        if shape.get('label') == 'lesion':
            shape['label'] = new_label
            modified = True
    if modified:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Updated: {json_path}")
        return True
    else:
        print(f"No 'lesion' label found in {json_path.name}")
        return False


# 指定 JSON 目录
json_root = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\temp_plantseg_labelme_new")

# 遍历所有 split 子目录（train, val, test）
for split in ['train', 'val', 'test']:
    split_dir = json_root / split
    if split_dir.exists():
        for json_file in split_dir.glob("*.json"):
            update_json_label(json_file)