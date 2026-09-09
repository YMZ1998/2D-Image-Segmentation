import json
from pathlib import Path

ROOT = Path(r"D:\data\OCT")

changed_files = 0
changed_labels = 0

for json_file in ROOT.rglob("*.json"):
    try:
        data = json.loads(json_file.read_text(encoding="utf-8-sig"))
    except Exception as e:
        print(f"[跳过] 无法读取: {json_file}，原因: {e}")
        continue

    # 只处理 LabelMe JSON
    if not isinstance(data, dict) or not isinstance(data.get("shapes"), list):
        continue

    file_changed = False

    for shape in data["shapes"]:
        if not isinstance(shape, dict):
            continue

        if shape.get("label") == "5":
            shape["label"] = "3"
            changed_labels += 1
            file_changed = True

    if file_changed:
        json_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        changed_files += 1
        print(f"[已修改] {json_file}")

print()
print(f"修改文件数: {changed_files}")
print(f"5 -> 3 标签数量: {changed_labels}")
