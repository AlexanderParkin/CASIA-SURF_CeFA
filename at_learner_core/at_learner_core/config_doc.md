# Каким должен быть config файл
Конфигурационный файл, подаваемый на вход `at_learner_core.trainer.Model` является `argparse.Namespace` объектом.
Можно создать объект `dict` и перегнать его в `Namespace` при помощи `at_learner_core.configs.dict_to_namespace`.

Конфиг файл имеет 7 логических ключевых частей: 
1. head_config
2. checkpoint_config
3. datalist_config
4. train_process_config
5. test_process_config
6. wrapper_config
7. logger_config

## head_config
* task_name: название задачи
* exp_name: название экспримента
```python
'head_config': {
            'task_name': 'Test task',
            'exp_name': 'Test exp 0',
}
```
## checkpoint_config
* out_path: путь до директории, где будут храниться чекпоинты, логи и т.д.
* save_frequency: частота сохранения чекпоинтов
```python
'checkpoint_config': {
            'out_path': None,
            'save_frequency': 1,
}
```
## datalist_config
Содержит конфигураци тренировочного и тестового списка.
* trainlist_config: содержит в себе путь до тренировочного списка и трансформы, которые применяются к данным
* testlist_config: аналогично trainlist_config, но в дороботке будет списком, котор
```python
'datalist_config': {
            'trainlist_config': {
                'datalist_path': '/path/to/train/list/',
                'transforms': tv.transforms.Compose([
                                tv.transforms.CenterCrop(224),
                                tv.transforms.RandomResizedCrop(size=112,
                                                                scale=(0.8, 1.0),
                                                                ratio=(0.9, 1.1111)),
                                tv.transforms.RandomHorizontalFlip(),
                                tv.transforms.ToTensor(),
                                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
            },
            'testlist_configs': {
                'datalist_path': '/path/to/test1/list/',
                'transforms': tv.transforms.Compose([
                                tv.transforms.CenterCrop(224),
                                tv.transforms.Resize(112),
                                tv.transforms.ToTensor(),
                                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

                }
```
## train_process_config
## test_process_config
## wrapper_config
## logger_config
## manual_seed
## resume