import os
import uuid
import shutil

class ModelService:
    def __init__(self):
        self.output_dir = "./outputs"
        os.makedirs(self.output_dir, exist_ok=True)

        # путь до твоего видео-заглушки
        self.stub_video_path = "video_2025-12-11_19-39-33.mp4"

        if not os.path.exists(self.stub_video_path):
            raise FileNotFoundError(
                f"Файл заглушки не найден: {self.stub_video_path}"
            )

    def load(self):
        print("[ModelService] Загружаем заглушку видеогенератора...")
        print(f"[ModelService] Используем видео: {self.stub_video_path}")

    def predict(self, image_bytes: bytes):
        # генерируем уникальное имя файла
        out_path = os.path.join(self.output_dir, f"{uuid.uuid4()}.mp4")

        # копируем заранее подготовленное видео как будто оно сгенерировано
        shutil.copy(self.stub_video_path, out_path)

        return out_path
