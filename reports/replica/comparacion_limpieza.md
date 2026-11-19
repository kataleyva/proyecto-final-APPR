# Réplica del artículo: con y sin nuestra limpieza

Mismo modelo, mismo preprocesamiento y mismas semillas ([0, 1, 2, 3, 4]). Solo cambian los datos:
2571 textos crudos (idénticos a los de los autores) contra 2531 textos limpios.

| Modelo | Accuracy sin limpieza | Accuracy con limpieza | Diferencia | p (Welch) | Autores |
|---|---|---|---|---|---|
| Random Forest + tf-idf | 0.775 ± 0.017 | 0.768 ± 0.023 | -0.008 | 0.551 | 0,802 (una sola partición 80/20, sin semilla) |
| BETO congelado + LSTM | 0.761 ± 0.027 | 0.702 ± 0.020 | -0.059 | 0.005 | 0,769 de media en 5 particiones; 0,80 en la mejor (la que reporta el artículo) |

Accuracy = media ± desviación estándar en las particiones. p < 0,05 sugiere una diferencia real; con 5 particiones es solo orientativo.

- Random Forest + tf-idf: en la versión sin limpieza, 66 noticias de test (sumando las particiones) tenían un texto idéntico en train.
- BETO congelado + LSTM: en la versión sin limpieza, 40 noticias de test (sumando las particiones) tenían un texto idéntico en train.
