calorie-prediction/
│
├── data/                          # Папка с данными (можно .gitignore)
│   ├── ingredients.csv
│   ├── dish.csv
│   └── images/                    # Папка с фотографиями
│
├── scripts/                       # Python файлы с кодом
│   ├── dataset.py                 # Загрузчики данных
│   ├── model.py                   # Архитектура нейросети
│   ├── train.py                   # Функция обучения
│   ├── utils.py                   # Вспомогательные функции
│   └── config.py                  # Конфигурация
│
├── notebooks/                     # (или просто файл в корне)
│   └── project_notebook.ipynb     # Jupyter с EDA и запуском
│
├── models/                        # Сохранённые модели
│   └── best_model.pth
│
├── README.md                      # Описание проекта
└── requirements.txt               # Зависимости
