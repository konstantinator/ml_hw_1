# ml_hw_1

В ходе работы над ДЗ было получена масса плохих ощущений от плохой подготовительной работы авторами
Неточности, разночтения ошибки, всё что мог и на что было желание пометил в ноутбуке ресёрча

Что сделано проведено EDA, обучено несколько моделей, проведены самые разнообразные преобразования над фичами
и построен пайплайн для быстрого использования модели в инференсе

В результате были получены хорошие метрики R2 0.9195 на тесте

Однако показатели бизнес метрики оставляют желать лучшего - всего 0.6

Возможно это связано с тем, что я логарифмировал целевую переменную, чтобы получить распределение 
более похожее на нормальное и линейной регрессии было бы легче учиться.

Мл модель была завёрнута в веб сервис

/predict_item
![0](https://github.com/konstantinator/ml_hw_1/blob/main/img/0.png)

/predict_items
![1](https://github.com/konstantinator/ml_hw_1/blob/main/img/1.png)

