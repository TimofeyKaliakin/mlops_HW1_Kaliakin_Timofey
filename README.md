# Калякин Тимофей 
# ДЗ 1
# Цель проекта
Создать воспроизводимый пайплайн обученя ML-модели с помощью MLflow и DVC для версионированя данных. 
Используемы датасет - Iris, обучается логистическая регрессия.

## Порядок запуска через терминал Mac
```bash
# brew install python@3.12 --- если отсутствует данная версия питона
python3.12 -m venv .venv && source .venv/bin/activate
git clone https://github.com/TimofeyKaliakin/mlops_HW1_Kaliakin_Timofey.git
cd mlops_HW1_Kaliakin_Timofey
pip install -r requirements.txt
dvc remote list  # опционально, чтобы проверить remote

dvc pull
dvc repro
mlflow ui --backend-store-uri sqlite:///mlflow.db  
```

- DVC-remote настроен на Yandex Object Storage (`s3://mlopshw123`); вместо экспорта ключей можно создать локальный `.dvc/config.local` (файл уже в `.gitignore`) с `access_key_id`/`secret_access_key`/`region`.

## Краткое описание пайплайна
- `prepare`: читает `data/raw/data.csv`, чистит пропуски/дубликаты, делит на train/test по `params.yaml`, сохраняет в `data/processed/`.
- `train`: обучает `StandardScaler + LogisticRegression` с параметрами из `params.yaml`, логирует параметры и метрику `accuracy` в MLflow, сохраняет `model.pkl`.

## UI MLflow
Backend: `sqlite:///mlflow.db`. Запуск UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db` (по умолчанию должен быть http://127.0.0.1:5000).
