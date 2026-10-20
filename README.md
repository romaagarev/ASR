# ASR

## Установка

```bash
python3.11 -m venv project_env
source project_env/bin/activate
pip install -r requirements.txt
pip install transformers==4.43.3
pip install --no-deps -r requirements_lm.txt
```

## Инференс

Датасет должен иметь такую структуру:

```text
dataset/
  audio/
    sample.wav
  transcriptions/
    sample.txt
```

Папка `transcriptions/` необязательна. Имена аудио и текстов должны совпадать без расширения.

Запуск с beam search и ByGPT5-small-en. Языковая модель скачивается автоматически:

```bash
python inference.py --config-name inference_deepspeech2 \
  input_dir=/path/to/dataset \
  inferencer.from_pretrained=/path/to/model.pth \
  inferencer.save_path=/path/to/predictions
```

Предсказания будут в `/path/to/predictions/custom/`, по одному текстовому файлу на запись.

CER и WER считаются отдельной командой:

```bash
python calc_metrics.py /path/to/dataset/transcriptions /path/to/predictions/custom
```

Скрипт выводит средние CER и WER по записям в процентах.

## Demo

Откройте demo.ipynb в Colab, укажите `DATASET_URL` и запустите все ячейки. В корне ZIP- или TAR-архива должны лежать `audio/` и, если есть разметка, `transcriptions/`.
