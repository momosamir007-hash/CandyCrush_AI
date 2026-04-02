# classifier_v2.py
"""
╔══════════════════════════════════════════════════╗
║ 🧠 CLIP Classifier v2 - يتعرف على 50+ عنصر ║
╚══════════════════════════════════════════════════╝
"""
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple, Dict, Optional
from candy_elements import (
    ALL_ELEMENTS,
    get_flat_descriptions,
    GameElement,
    ElementCategory
)
import time

class CandyCrushClassifierV2:
    """
    مصنف شامل لجميع عناصر كاندي كراش
    يتعرف على 50+ نوع مختلف
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = None,
        active_categories: List[ElementCategory] = None
    ):
        """
        Args:
            model_name: نموذج CLIP
            device: "cuda" أو "cpu"
            active_categories: الفئات المراد كشفها None = كل الفئات
        """
        print("🔄 تحميل CLIP v2 الشامل...")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # ═══ تصفية العناصر حسب الفئات المطلوبة ═══
        if active_categories:
            self.active_elements = {
                k: v for k, v in ALL_ELEMENTS.items()
                if v.category in active_categories
            }
        else:
            self.active_elements = ALL_ELEMENTS

        # ═══ تجهيز الأوصاف ═══
        self.descriptions = []
        self.desc_to_id = []
        for elem_id, elem in self.active_elements.items():
            for desc in elem.clip_descriptions:
                self.descriptions.append(desc)
                self.desc_to_id.append(elem_id)

        # ═══ تشفير النصوص مسبقاً ═══
        self._precompute_text_features()
        print(f"✅ جاهز! {len(self.active_elements)} عنصر")
        print(f"   {len(self.descriptions)} وصف نصي مشفر")
        print(f"   الجهاز: {self.device}\n")

    def _precompute_text_features(self):
        """تشفير كل النصوص دفعة واحدة"""
        # تقسيم لدفعات إذا كانت كثيرة
        batch_size = 64
        all_features = []
        for i in range(0, len(self.descriptions), batch_size):
            batch = self.descriptions[i:i + batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            with torch.no_grad():
                feats = self.model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_features.append(feats)
        self.text_features = torch.cat(all_features, dim=0)

    @torch.no_grad()
    def classify_cell(
        self,
        cell_image,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        تصنيف خلية واحدة
        Returns: قائمة بأعلى top_k نتائج [(element_id, confidence), ...]
        """
        if isinstance(cell_image, np.ndarray):
            cell_image = Image.fromarray(
                cell_image[:, :, ::-1]
                if cell_image.shape[2] == 3 and cell_image.dtype == np.uint8
                else cell_image
            )

        inputs = self.processor(
            images=cell_image,
            return_tensors="pt"
        ).to(self.device)

        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # التشابه
        similarities = (image_features @ self.text_features.T).squeeze()

        # تجميع حسب العنصر (متوسط أوصافه)
        element_scores = {}
        element_counts = {}
        for idx, (score, elem_id) in enumerate(zip(similarities, self.desc_to_id)):
            s = score.item()
            if elem_id not in element_scores:
                element_scores[elem_id] = 0
                element_counts[elem_id] = 0
            element_scores[elem_id] += s
            element_counts[elem_id] += 1

        avg_scores = {
            k: element_scores[k] / element_counts[k]
            for k in element_scores
        }

        # Softmax للثقة
        scores_array = np.array(list(avg_scores.values()))
        keys = list(avg_scores.keys())
        exp_scores = np.exp((scores_array - scores_array.max()) * 15)
        softmax = exp_scores / exp_scores.sum()

        # ترتيب
        ranked = sorted(zip(keys, softmax), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    @torch.no_grad()
    def classify_batch(
        self,
        cell_images: List,
        batch_size: int = 32
    ) -> List[Tuple[str, float]]:
        """تصنيف دفعة كاملة من الخلايا"""
        all_results = []
        for i in range(0, len(cell_images), batch_size):
            batch_imgs = cell_images[i:i + batch_size]

            # تحويل لـ PIL
            pil_batch = []
            for img in batch_imgs:
                if isinstance(img, np.ndarray):
                    pil_batch.append(
                        Image.fromarray(img[:, :, ::-1])
                        if len(img.shape) == 3
                        else Image.fromarray(img)
                    )
                else:
                    pil_batch.append(img)

            inputs = self.processor(
                images=pil_batch,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarities = image_features @ self.text_features.T

            for sim in similarities:
                element_scores = {}
                element_counts = {}
                for idx, (score, elem_id) in enumerate(zip(sim, self.desc_to_id)):
                    s = score.item()
                    if elem_id not in element_scores:
                        element_scores[elem_id] = 0
                        element_counts[elem_id] = 0
                    element_scores[elem_id] += s
                    element_counts[elem_id] += 1

                avg = {
                    k: element_scores[k] / element_counts[k]
                    for k in element_scores
                }
                scores_arr = np.array(list(avg.values()))
                keys = list(avg.keys())
                exp_s = np.exp((scores_arr - scores_arr.max()) * 15)
                sm = exp_s / exp_s.sum()
                best_idx = np.argmax(sm)
                all_results.append((keys[best_idx], float(sm[best_idx])))

        return all_results

    def classify_grid(
        self,
        image: np.ndarray,
        rows: int = 9,
        cols: int = 9,
        padding: float = 0.12,
        board_region: Tuple[int, int, int, int] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[List]]:
        """
        تصنيف اللوحة الكاملة
        Returns:
            grid: مصفوفة أسماء العناصر
            confidence: مصفوفة نسب الثقة
            details: تفاصيل (top-3 لكل خلية)
        """
        h, w = image.shape[:2]
        if board_region:
            bx, by, bw, bh = board_region
            board = image[by:by+bh, bx:bx+bw]
        else:
            board = image
            bh, bw = h, w

        cell_h = bh // rows
        cell_w = bw // cols
        pad_y = int(cell_h * padding)
        pad_x = int(cell_w * padding)

        # قص كل الخلايا
        cells = []
        positions = []
        for r in range(rows):
            for c in range(cols):
                y1 = r * cell_h + pad_y
                y2 = (r + 1) * cell_h - pad_y
                x1 = c * cell_w + pad_x
                x2 = (c + 1) * cell_w - pad_x
                cell = board[y1:y2, x1:x2]
                if cell.size > 0:
                    from cv2 import resize
                    cell = resize(cell, (72, 72))
                    cells.append(cell)
                else:
                    cells.append(np.zeros((72, 72, 3), dtype=np.uint8))
                positions.append((r, c))

        print(f"🔍 تصنيف {len(cells)} خلية...")
        start = time.time()
        results = self.classify_batch(cells)
        elapsed = time.time() - start
        print(f"⏱️ تم في {elapsed:.2f}s")

        # بناء المصفوفات
        grid = np.full((rows, cols), "empty", dtype=object)
        conf = np.zeros((rows, cols))
        details = [[None for _ in range(cols)] for _ in range(rows)]

        for (r, c), (elem_id, confidence) in zip(positions, results):
            grid[r, c] = elem_id
            conf[r, c] = confidence

        return grid, conf, details

    def get_element_info(self, elem_id: str) -> Optional[GameElement]:
        """معلومات عنصر"""
        return ALL_ELEMENTS.get(elem_id)

    def get_element_emoji(self, elem_id: str) -> str:
        elem = ALL_ELEMENTS.get(elem_id)
        return elem.emoji if elem else '❓'
