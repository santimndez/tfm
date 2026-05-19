# Estimación de trayectorias en tenis de mesa

Estructura del repositorio:

- `src/`: código fuente de los programas y notebooks de python del proyecto:
  - `src/utils/`: paquete en python de utilidades:
    - `src/utils/ball_detection.py`: utilidades para la detección de la pelota en vídeos a partir de un modelo de segmentación.
    - `src/utils/binocular.py`: funciones para cálculos de geometría proyectiva con una o dos cámaras.
    - `src/utils/sync.py`: utilidades para la estimación del desfase y sincronización de dos vídeos de una misma escena de tenis de mesa.
    - `src/utils/filter.py`: módulo para el tracking de la pelota estimando trayectoria, velocidad y efecto mediante UKF y RTS smoother.
    - `src/utils/tt_animation.py`: módulo para generar animaciones que muestran la trayectoria, velocidad y efecto de la pelota en una escena.
  - `src/files`: carpeta con archivos para reproducir los resultados obtenidos.
    - `src/calibration`: carpeta con archivos para la calibración intrínseca y extrínseca de la cámara.
      - `src/calibration/refs.txt: archivo con las referencias para la calibración extrínseca de la cámara correspondientes a los vídeos utilizados.
    - `src/detection`: carpeta para guardar archivos con las detecciones de la pelota en distintos vídeos.
    - `src/segmentation`: carpeta para guardar las máscaras de segmentación obtenidas para algunos vídeos.
    - `src/models`: carpeta para guardar los modelos de segmentación utilizados.
      - `src/models/download_models.py`: script para descargar los modelos utilizados.
      - `src/models/model_ids.txt`: contiene los ID's de los modelos, guardados en Google Drive.
    - `src/videos`: carpeta para guardar los vídeos utilizados.
      - `src/videos/download_videos.py`: script para descargar los vídeos utilizados.
      - `src/videos/video_ids.txt`: contiene los ID's de los vídeos, guardados en Google Drive.
  - `src/notebooks`: contiene los notebooks para la obtención de un modelo de segmentación YOLO11n-seg mediante destilación de SAM2 y para la segmentación de la pelota con SAM3.
  - `src/synchronize_video.py`: sincroniza dos vídeos de una misma escena de tenis de mesa minimizando el error de reproyección de las detecciones de la pelota. Después, estima la trayectoria, velocidad y efecto de la pelota a lo largo de la escena.
  - `src/show_epilines.py`: Muestra la detección de la pelota en dos vídeos y las líneas epipolares correspondientes. Útil para depurar la sincronización de vídeos.
  - `src/rally_animation.py`: Genera una animación con la trayectoria, velocidad y efecto de la pelota a partir de las estimaciones obtenidas con el módulo `filter.py`.
  - `src/get_positions_from_masks.py`: obtiene las posiciones de la pelota detectadas en un vídeo a partir de las máscaras de segmentación guardadas en una carpeta.
  - `src/draw_segmentation.py`: dibuja la segmentación de la pelota obtenida mediante un modelo de segmentación, o bien a partir de las máscaras de segmentación guardadas en una carpeta.
  - `src/select_points.py`: permite seleccionar puntos en un frame seleccionado de un vídeo y mostrar sus coordenadas haciendo clic.
