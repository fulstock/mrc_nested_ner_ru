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

## Ноутбук

Помимо использования скриптов, можно воспользоваться ноутбуком `Main.ipynb`. В нём можно задать все параметры обучения, провести само обучение, тестирование и оценку вывода. См. `Main.ipynb` за подробностями. 

## Установка
`pip install -r requirements.txt`

Проект построен на [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning).
Все подробности о параметрах обучения можно узнать в [документации pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/).

## Данные

Перед началом использования требуется переформатировать исходные данные. Они должны быть подготовлены в формате, предоставлямом системой [**BRAT**](https://brat.nlplab.org/standoff.html). Конвертация данных в формат `mrc-json`, принимаемый моделью, производится с помощью запуска скрипта `datasets\brat2mrc.py`. См. файл за подробностями. 

Для использования и применения какой-либо подсказки для модели, вместо `datasets\brat2mrc.py` требуется воспользоваться соответствующим скриптом из каталога `prompts`. Возможные подсказки перечислены там же. 

## Обучение
Обучение - запуск скрипта `trainer.py` с соответствующими параметрами.

Для минимального запуска требуется указать директорию с моделью и директорию с данными <br>
через `--bert_config_dir` и `--data_dir` соответственно.

Данные должны быть подготовлены в формате, предоставляемом системой [**BRAT**](https://brat.nlplab.org/standoff.html).

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

Дополнительно присутствует система вывода модели `inference.py`. Данные на входе и выходе - в формате `mrc-json`. См. файл за подробностями.

Также в каталоге `ckpt` добавлены ссылки на обученные модели на разных подсказках на данных `NEREL-bio`. См. `ckpt\links.md`.

---

Based upon
**A Unified MRC Framework for Named Entity Recognition** <br>
Xiaoya Li, Jingrong Feng, Yuxian Meng, Qinghong Han, Fei Wu and Jiwei Li<br>
In ACL 2020. ([Article](https://arxiv.org/abs/1910.11476)) and corresponding [repo](https://github.com/ShannonAI/mrc-for-flat-nested-ner),
<br>
<br>
as well as
**Adaptation of Deep Bidirectional Multilingual Transformers for Russian Language** <br>
Yuri Kuratov, Mikhail Arkhipov ([Article](https://arxiv.org/abs/1905.07213))
and the [model](https://huggingface.co/DeepPavlov/rubert-base-cased).<br>

## Notebook

In addition to using scripts, you can use the notebook `Main.ipynb`. There you can set all the training parameters, run the training itself, test and evaluate the output. See `Main.ipynb` for details.

## Installation
`pip install -r requirements.txt`

The project is built on [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning).
For full details on training parameters, see the [pytorch-lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/).

## Data

Before usage, you need to reformat the original data. They must be prepared in the format provided by the [**BRAT**](https://brat.nlplab.org/standoff.html) system. Converting data to the `mrc-json` format accepted by the model is done by running the `datasets\brat2mrc.py` script. See the file for details.

To use and apply any prompts for the model, instead of `datasets\brat2mrc.py`, you need to use the appropriate script from the `prompts` directory. Possible prompts are listed there.

## Training
Run the `trainer.py` script with the appropriate parameters.

For a minimal launch, you need to specify a directory with a model and a directory with data <br>
via `--bert_config_dir` and `--data_dir` respectively.

Data must be prepared in the format provided by the [**BRAT**](https://brat.nlplab.org/standoff.html) system.

Example:<br>
`python .\trainer.py --bert_config_dir .\bert --data_dir .\dataset`<br>
Another example: <br>
`python .\trainer.py --pretrained_checkpoint <path to checkpoint> --bert_config_dir .\bert `<br>`--data_dir .\data --batch_size 4 --max_epochs 1 --val_check_interval 0.25` - start training from checkpoint based on bert and data with minibatch size of 4, one epoch and validation every 0.25 batches of one epoch.

All other parameters (output directory, model hyperparameters, ...) can be found by calling `--help`.

## Testing
`trainer.py` is automatically validated on dev subset every `val_check_interval` epochs, <br>
and stores the best `k` checkpoints in `default_root_dir`.

To run test on the test subset, run the `tester.py` script with the same command line options as for `trainer.py`.

It is also possible to look at the output of the model and check it on the original markup: see the `test_dataset.py` script.

Additionally, there is an `inference.py` model inference system. The input and output data is in `mrc-json` format. See file for details.

Also in the `ckpt` directory, links to trained models with different prompts on the `NEREL-bio` data have been added. See `ckpt\links.md`.