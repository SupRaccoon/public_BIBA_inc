# public_BIBA_inc
Public version without any data. 

# Моделирование движений B- и T-клеток
Репозиторий для написания вычислительной части проекта по моделированию движения B- и T- клеток.

## Quick start

Расчеты реализованы как скрипт, главный файл — *main.py*, у него есть несколько ключей:
- `-iters` — определяет количество итераций (поворотов), которые нужно рассчитать. __Обязательный параметр__.
- `-nc` — определяет количество клеток, для которых ведётся расчет. __Обязательный параметр__.
- `-data_path` — определяет путь к бинарным данным карты, а именно к папке с данными. 
  Важно, что в папке названия бинарных файлов должны соответствовать названиям зон, так как эти названия фигурируют
  в качестве ключей в коде. __Необязательный параметр__, так как по умолчанию ищет папку _data_map/_ в текущей рабочей
  директории.
- `-out` — определяет путь для записи результатов расчетов. __Необязательный параметр__, так как по умолчанию ищет папку
  *calc_output*, если не находит её, то пишет результаты расчетов в текущую рабочую директорию. Результатам расчетов
  всегда даёт имя _result_, ха-ха и что вы мне сделаете? Ничего, правильно.
- `-mod_vel` — определяет путь к данным, содержащим записанные значения скорости (сразу в пкс/сек). 
  __Необязательный параметр__, по умолчанию содержит значение 0.046 pxl\min ~= 8 mkm\min.
- `-mod_coord` — определяет путь к данным, содержащим записанные значения начальных координат. 
  __Необязательный параметр__, по умолчанию содержит значение `np.array([250.0, 250.0, 50.0])`.
- `-ss` — определяет надо или нет делать начальные шаги ДО итераций. 
  __Необязательный параметр__, по умолчанию `false`, получает `true`, если передан.
- `-no_pool` — определяет необходимость использования нескольких процессов при вычислениях. 
  __Необязательный параметр__, по умолчанию `true`, получает `false`, если передан.
  
## Структура репозитория

```angular2html
│   .gitignore
│   main.py                            — главный исполняемый файл расчетов
│   README.md                          — непосредственно readme
└───cells                              — основные классы, используемые в проекте
    ├───angle.py                       - класс углов
    └───cell.py                        - класс клеток
