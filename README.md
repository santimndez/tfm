# Estimación de trayectorias en tenis de mesa

Estructura del repositorio:
- `src/`: código fuente de los programas y notebooks de python del proyecto:
  - `src/utils/`: paquete en python de utilidades:
    - `src/utils/ball_detection.py`: utilidades para la detección de la pelota en vídeos a partir de un modelo de segmentación.
    - `src/utils/binocular.py`: funciones para cálculos de geometría proyectiva con una o dos cámaras.
    - `src/utils/sync.py`: utilidades para la estimación del desfase y sincronización de dos vídeos de una misma escena de tenis de mesa. 
  - `src/synchronize_video.py`: script para sincronizar dos vídeos estéreo minimizando el error de reproyección de la segmentación de la pelota.
  - `src/show_epilines.py`: Muestra la detección de la pelota en dos vídeos y las líneas epipolares correspondientes. Útil para depurar la sincronización de vídeos.
