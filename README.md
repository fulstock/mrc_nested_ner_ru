# A Unified MRC Framework for Nested Named Entity Recognition (RU rework)

Основано на
**A Unified MRC Framework for Named Entity Recognition** <br>
Xiaoya Li, Jingrong Feng, Yuxian Meng, Qinghong Han, Fei Wu and Jiwei Li<br>
In ACL 2020. ([Статья](https://arxiv.org/abs/1910.11476)) и соответствующем [репозитории](https://github.com/ShannonAI/mrc-for-flat-nested-ner), 
<br>
<br>
а также
**Adaptation of Deep Bidirectional Multilingual Transformers for Russian Language** <br>
Yuri Kuratov, Mikhail Arkhipov ([Статья](https://arxiv.org/abs/1905.07213))
и самой [модели](https://huggingface.co/DeepPavlov/rubert-base-cased).<br>

## Установка
`pip install -r requirements.txt`

Проект построен на [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning).
Все подробности о параметрах обучения можно узнать в [документации pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/).

## Обучение
Обучение - запуск скрипта `trainer.py` с соответствующими параметрами.

Для минимального запуска требуется указать директорию с моделью и директорию с данными <br>
через `--bert_config_dir` и `--data_dir` соответственно.

Данные должны быть подготовлены в формате, предоставляемом системой [**BRAT**](https://brat.nlplab.org/).

Пример:<br>
`python .\trainer.py --bert_config_dir .\bert --data_dir .\dataset`<br>
Еще пример:<br>
`python .\trainer.py --pretrained_checkpoint <путь к чекпойнту> --bert_config_dir .\bert `<br>`--data_dir .\data --batch_size 4 --max_epochs 1 --val_check_interval 0.25` - запуск обучения с чекпойнта на основе bert и данных data с размером минибатчей в 4, одной эпохой и валидирования каждые 0.25 батчей одной эпохи. 

Все остальные параметры (выходную директорию, гиперпараметры модели, ...) можно узнать, <br>вызвав справку `--help`.

## Тестирование
`trainer.py` автоматически валидируется на dev выборке каждые `val_check_interval` эпох, <br>
и сохраняет лучшие `k` чекпойнтов в `default_root_dir`.

Для тестирования на test следует запустить скрипт `tester.py` с теми же параметрами командной строки, как и для `trainer.py`.

Также возможно взглянуть на вывод модели и сверить её с исходной разметкой: см. скрипт `test_dataset.py`.
