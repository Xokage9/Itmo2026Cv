import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import h5py
import numpy as np
import yaml


def _decode_filename(h5_file: h5py.File, name_ref) -> str:
    data = h5_file[name_ref][()]
    data = np.array(data).reshape(-1)
    return ''.join(chr(int(x)) for x in data)


def _read_attr_values(h5_file: h5py.File, bbox_group: h5py.Group, attr_name: str) -> List[float]:
    attr = bbox_group[attr_name]
    if isinstance(attr, h5py.Dataset):
        values = attr[()]
        values = np.array(values)
        if values.dtype == np.object_ or values.dtype.kind == 'O':
            out = []
            for ref in values.reshape(-1):
                ref_data = h5_file[ref][()]
                out.append(float(np.array(ref_data).reshape(-1)[0]))
            return out
        flat = values.reshape(-1)
        return [float(v) for v in flat]

    out = []
    for key in attr.keys():
        out.append(float(np.array(attr[key][()]).reshape(-1)[0]))
    return out


def svhn_to_yolo_labels(split_dir: Path) -> Dict[str, float]:
    mat_path = split_dir / 'digitStruct.mat'
    if not mat_path.exists():
        raise FileNotFoundError(f'Missing {mat_path}')

    total = 0
    written = 0

    with h5py.File(mat_path, 'r') as f:
        ds = f['digitStruct']
        names = ds['name']
        bboxes = ds['bbox']

        for idx in range(len(names)):
            total += 1
            name_ref = names[idx][0]
            filename = _decode_filename(f, name_ref)
            image_path = split_dir / filename
            if not image_path.exists():
                continue

            bbox_ref = bboxes[idx][0]
            bbox_group = f[bbox_ref]

            left = _read_attr_values(f, bbox_group, 'left')
            top = _read_attr_values(f, bbox_group, 'top')
            width = _read_attr_values(f, bbox_group, 'width')
            height = _read_attr_values(f, bbox_group, 'height')

            x1 = min(left)
            y1 = min(top)
            x2 = max(l + w for l, w in zip(left, width))
            y2 = max(t + h for t, h in zip(top, height))

            image = cv2.imread(str(image_path))
            if image is None:
                continue
            h, w = image.shape[:2]

            x1 = max(0.0, min(x1, w - 1.0))
            y1 = max(0.0, min(y1, h - 1.0))
            x2 = max(x1 + 1.0, min(x2, float(w)))
            y2 = max(y1 + 1.0, min(y2, float(h)))

            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            label_path = image_path.with_suffix('.txt')
            with open(label_path, 'w', encoding='utf-8') as lf:
                lf.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
            written += 1

    return {
        'split': split_dir.name,
        'images_indexed': total,
        'labels_written': written,
    }


def create_dataset_yaml(project_root: Path) -> Path:
    data = {
        'path': str(project_root.resolve()).replace('\\', '/'),
        'train': 'train',
        'val': 'test',
        'test': 'test',
        'names': {0: 'number'},
    }
    yaml_path = project_root / 'svhn_number.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return yaml_path


def prepare(project_root: Path) -> Dict:
    stats = {
        'train': svhn_to_yolo_labels(project_root / 'train'),
        'test': svhn_to_yolo_labels(project_root / 'test'),
    }
    yaml_path = create_dataset_yaml(project_root)
    stats['data_yaml'] = str(yaml_path)

    out_path = project_root / 'lab5_dataset_stats.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def train(project_root: Path, epochs: int = 8, imgsz: int = 416, batch: int = 64):
    os.environ.setdefault('YOLO_CONFIG_DIR', str((project_root / '.yolo_cfg').resolve()))
    from ultralytics import YOLO

    data_yaml = project_root / 'svhn_number.yaml'
    model = YOLO('yolov8n.pt')
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        workers=4,
        project=str(project_root / 'runs' / 'lab5'),
        name='svhn_yolov8n',
        exist_ok=True,
        patience=4,
        optimizer='AdamW',
        lr0=1e-3,
        cos_lr=True,
    )


def validate(project_root: Path, weights_path: Path = None) -> Dict:
    os.environ.setdefault('YOLO_CONFIG_DIR', str((project_root / '.yolo_cfg').resolve()))
    from ultralytics import YOLO

    if weights_path is None:
        weights_path = project_root / 'runs' / 'lab5' / 'svhn_yolov8n' / 'weights' / 'best.pt'
    model = YOLO(str(weights_path))
    metrics = model.val(data=str(project_root / 'svhn_number.yaml'), split='val', verbose=False)

    out = {
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'map50': float(metrics.box.map50),
        'map50_95': float(metrics.box.map),
    }

    out_path = project_root / 'lab5_metrics.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def _xywhn_to_xyxy(label_line: str, w: int, h: int) -> Tuple[float, float, float, float]:
    _, cx, cy, bw, bh = [float(x) for x in label_line.strip().split()]
    x1 = (cx - bw / 2.0) * w
    y1 = (cy - bh / 2.0) * h
    x2 = (cx + bw / 2.0) * w
    y2 = (cy + bh / 2.0) * h
    return x1, y1, x2, y2


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_mean_iou(project_root: Path, conf: float = 0.25, max_images: int = 3000) -> Dict:
    os.environ.setdefault('YOLO_CONFIG_DIR', str((project_root / '.yolo_cfg').resolve()))
    from ultralytics import YOLO

    weights_path = project_root / 'runs' / 'lab5' / 'svhn_yolov8n' / 'weights' / 'best.pt'
    model = YOLO(str(weights_path))

    image_paths = sorted((project_root / 'test').glob('*.png'))
    if max_images and max_images < len(image_paths):
        image_paths = image_paths[:max_images]

    ious = []
    matched = 0

    for img_path in image_paths:
        label_path = img_path.with_suffix('.txt')
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(label_path, 'r', encoding='utf-8') as f:
            gt_line = f.readline().strip()
        if not gt_line:
            continue
        gt = _xywhn_to_xyxy(gt_line, w, h)

        pred = model.predict(source=str(img_path), conf=conf, iou=0.5, verbose=False, max_det=1)
        if len(pred) == 0 or pred[0].boxes is None or len(pred[0].boxes) == 0:
            continue

        box = pred[0].boxes.xyxy[0].cpu().numpy().tolist()
        pred_box = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        iou_v = _iou(gt, pred_box)
        ious.append(iou_v)
        matched += 1

    result = {
        'mean_iou': float(np.mean(ious)) if ious else 0.0,
        'matched_predictions': int(matched),
        'evaluated_images': int(len(image_paths)),
    }

    out_path = project_root / 'lab5_iou.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def infer_city_images(project_root: Path, conf: float = 0.25):
    os.environ.setdefault('YOLO_CONFIG_DIR', str((project_root / '.yolo_cfg').resolve()))
    from ultralytics import YOLO

    weights_path = project_root / 'runs' / 'lab5' / 'svhn_yolov8n' / 'weights' / 'best.pt'
    model = YOLO(str(weights_path))

    source_dir = project_root / 'images_lab5'
    return model.predict(
        source=str(source_dir),
        conf=conf,
        iou=0.5,
        save=True,
        project=str(project_root / 'runs' / 'lab5'),
        name='street_infer',
        exist_ok=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'train', 'validate', 'iou', 'infer', 'all'])
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--max-images', type=int, default=3000)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    if args.action in {'prepare', 'all'}:
        stats = prepare(project_root)
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.action in {'train', 'all'}:
        train(project_root, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)

    if args.action in {'validate', 'all'}:
        metrics = validate(project_root)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.action in {'iou', 'all'}:
        iou = compute_mean_iou(project_root, max_images=args.max_images)
        print(json.dumps(iou, ensure_ascii=False, indent=2))

    if args.action in {'infer', 'all'}:
        infer_city_images(project_root)
        print('Inference on images_lab5 completed.')


if __name__ == '__main__':
    main()
