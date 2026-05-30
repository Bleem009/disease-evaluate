import json
import os
import glob
from PIL import Image


def plantseg_to_labelme(image_root_dir, json_dir, output_root_dir):
    """
    将多个PlantSeg COCO格式（train.json, test.json, val.json）转为LabelMe格式。
    图片可能分布在image_root_dir下的多个子文件夹中（按类别），但JSON只记录文件名。
    函数会递归搜索所有图片，然后根据JSON中的划分生成LabelMe文件，并保存到output_root_dir下的对应子文件夹。
    只生成图片实际存在的标注文件。

    Args:
        image_root_dir: 图片根目录，其下可能有多个子文件夹（如不同病害类别），包含图片文件。
        json_dir: COCO JSON文件所在目录，包含train.json, test.json, val.json。
        output_root_dir: 输出LabelMe JSON的根目录，函数会在其中创建train, test, val子文件夹，
                         并将对应划分的LabelMe文件放入其中。
    """
    os.makedirs(output_root_dir, exist_ok=True)

    # 递归搜索所有图片文件，建立文件名到完整路径的映射
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_root_dir, '**', ext), recursive=True))

    filename_to_path = {}
    for path in image_paths:
        filename = os.path.basename(path)
        if filename in filename_to_path:
            print(f"警告: 发现重复文件名 '{filename}'，将使用第一个找到的路径。")
        else:
            filename_to_path[filename] = path

    # 定义需要处理的划分
    splits = ['train', 'val', 'test']

    for split in splits:
        json_path = os.path.join(json_dir, f"{split}.json")
        if not os.path.exists(json_path):
            print(f"警告: JSON文件 {json_path} 不存在，跳过{split}划分。")
            continue

        # 加载COCO数据
        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        # 构建图像ID到图像信息的映射
        id_to_image = {img['id']: img for img in coco_data['images']}

        # 按图像ID分组标注
        image_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)

        # 准备输出子文件夹
        output_split_dir = os.path.join(output_root_dir, split)
        os.makedirs(output_split_dir, exist_ok=True)

        # 遍历该划分中的所有图像
        for img_info in coco_data['images']:
            img_name = img_info['file_name']
            img_id = img_info['id']

            # 检查图片文件是否存在，若不存在则跳过（不生成标注）
            if img_name not in filename_to_path:
                print(f"信息: 图片文件 '{img_name}' 未在图片目录中找到，跳过生成。")
                continue

            anns = image_annotations.get(img_id, [])
            if not anns:
                # 如果没有标注，可以选择生成空标注文件，或者跳过（这里选择跳过）
                # 如需生成空标注，请注释掉下一行
                continue

            # 构建LabelMe数据（尺寸直接从JSON获取，避免读取图片）
            labelme_data = {
                "version": "5.11.2",
                "flags": {},
                "shapes": [],
                "imagePath": img_name,
                "imageData": None,
                "imageHeight": img_info['height'],
                "imageWidth": img_info['width']
            }

            for ann in anns:
                segmentation = ann['segmentation']
                if not isinstance(segmentation, list) or len(segmentation) == 0:
                    continue
                poly = segmentation[0] if isinstance(segmentation[0], list) else segmentation
                points = [[poly[i], poly[i + 1]] for i in range(0, len(poly), 2)]
                if len(points) < 3:
                    continue
                labelme_data['shapes'].append({
                    "label": "lesion",
                    "points": points,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {},
                    "mask": None
                })

            # 保存
            output_path = os.path.join(output_split_dir, os.path.splitext(img_name)[0] + '.json')
            with open(output_path, 'w') as f:
                json.dump(labelme_data, f, indent=2)

            print(f"✓ {split}/{img_name}: {len(labelme_data['shapes'])} 个病灶")


def merge_two_labelme(leaf_labelme_dir, lesion_labelme_dir, output_dir):
    """合并叶片和病灶标注（保持不变）"""
    os.makedirs(output_dir, exist_ok=True)

    leaf_files = {os.path.basename(p): p for p in glob.glob(os.path.join(leaf_labelme_dir, '*.json'))}
    lesion_files = {os.path.basename(p): p for p in glob.glob(os.path.join(lesion_labelme_dir, '*.json'))}

    all_files = set(leaf_files.keys()) | set(lesion_files.keys())

    for filename in all_files:
        shapes = []
        image_info = None

        if filename in leaf_files:
            with open(leaf_files[filename], 'r') as f:
                leaf_data = json.load(f)
            shapes.extend(leaf_data.get('shapes', []))
            image_info = {
                'imagePath': leaf_data['imagePath'],
                'imageHeight': leaf_data['imageHeight'],
                'imageWidth': leaf_data['imageWidth']
            }

        if filename in lesion_files:
            with open(lesion_files[filename], 'r') as f:
                lesion_data = json.load(f)
            shapes.extend(lesion_data.get('shapes', []))
            if image_info is None:
                image_info = {
                    'imagePath': lesion_data['imagePath'],
                    'imageHeight': lesion_data['imageHeight'],
                    'imageWidth': lesion_data['imageWidth']
                }

        merged = {
            "version": "5.11.2",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_info['imagePath'],
            "imageData": None,
            "imageHeight": image_info['imageHeight'],
            "imageWidth": image_info['imageWidth']
        }

        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(merged, f, indent=2)

        leaf_count = sum(1 for s in shapes if s['label'] == 'leaf')
        lesion_count = sum(1 for s in shapes if s['label'] == 'lesion')
        print(f"✓ {filename}: {leaf_count} 叶片 + {lesion_count} 病灶 = {len(shapes)} 总计")


def labelme_to_coco(labelme_dir, image_dir, output_coco_path):
    """将合并后的LabelMe转为COCO格式（保持不变）"""
    labelme_files = glob.glob(os.path.join(labelme_dir, '*.json'))

    coco_data = {
        "info": {
            "description": "Merged leaf and lesion dataset",
            "version": "1.0",
            "year": 2024
        },
        "licenses": [{"id": 1, "name": "", "url": ""}],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "lesion", "supercategory": "disease"},
            {"id": 2, "name": "leaf", "supercategory": "plant"}
        ]
    }

    image_id = 1
    annotation_id = 1

    for labelme_path in labelme_files:
        with open(labelme_path, 'r') as f:
            data = json.load(f)

        img_name = data['imagePath']

        coco_data['images'].append({
            "id": image_id,
            "file_name": img_name,
            "height": data['imageHeight'],
            "width": data['imageWidth'],
            "license": 1,
            "date_captured": ""
        })

        for shape in data.get('shapes', []):
            if shape['shape_type'] != 'polygon':
                continue

            points = shape['points']
            label = shape['label']

            if label == 'lesion':
                category_id = 1
            elif label in ['leaf', '叶片', 'leaf_blade']:
                category_id = 2
            else:
                print(f"  跳过未知标签 '{label}' in {img_name}")
                continue

            segmentation = []
            for pt in points:
                segmentation.extend([float(pt[0]), float(pt[1])])

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

            coco_data['annotations'].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [segmentation],
                "area": bbox[2] * bbox[3],
                "bbox": bbox,
                "iscrowd": 0
            })
            annotation_id += 1

        print(f"✓ {img_name}: {len(data.get('shapes', []))} 个标注")
        image_id += 1

    with open(output_coco_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"COCO格式已保存: {output_coco_path}")
    print(f"图像数: {len(coco_data['images'])}")
    print(f"标注数: {len(coco_data['annotations'])}")
    print(f"  - 病灶(lesion): {sum(1 for ann in coco_data['annotations'] if ann['category_id'] == 1)}")
    print(f"  - 叶片(leaf): {sum(1 for ann in coco_data['annotations'] if ann['category_id'] == 2)}")


# ==================== 使用流程 ====================

if __name__ == "__main__":
    # 步骤1: 将PlantSeg COCO（分train/test/val）转为LabelMe（病灶）
    print("=" * 50)
    print("步骤1: PlantSeg转LabelMe（病灶）——支持图片散落在多个子文件夹")
    print("=" * 50)

    plantseg_to_labelme(
        image_root_dir=r'C:\Users\86159\Desktop\毕设\数据集\images',  # 该目录下可按类别存放多个子文件夹，图片均在内部
        json_dir=r'C:\Users\86159\Desktop\毕设\数据集\json',  # 存放 train.json, test.json, val.json 的目录
        output_root_dir=r'C:\Users\86159\PycharmProjects\disease_evaluation\temp_plantseg_labelme'
    )

    # 步骤2: 合并两个LabelMe（你的叶片 + PlantSeg病灶）
    # print("\n" + "=" * 50)
    # print("步骤2: 合并LabelMe标注")
    # print("=" * 50)
    #
    # merge_two_labelme(
    #     leaf_labelme_dir='my_leaf_annotations/',  # 你的叶片标注文件夹
    #     lesion_labelme_dir='temp_plantseg_labelme/',  # 上一步生成的病灶标注
    #     output_dir='merged_labelme/'  # 合并后的输出
    # )
    #
    # # 步骤3: 合并后的LabelMe → COCO
    # print("\n" + "=" * 50)
    # print("步骤3: 转为COCO格式")
    # print("=" * 50)
    #
    # labelme_to_coco(
    #     labelme_dir='merged_labelme/',
    #     image_dir='images/',
    #     output_coco_path='final_dataset.json'
    # )