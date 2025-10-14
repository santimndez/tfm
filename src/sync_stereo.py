import subprocess
import argparse
import cv2 as cv

# Argumentos de la línea de comandos
parser = argparse.ArgumentParser(description='Sincronizar dos vídeos estéreo')
parser.add_argument('-i', '--input', metavar='video_files', type=str, nargs=2, help='Rutas de los dos vídeos a sincronizar')
parser.add_argument('--offset', metavar='offset', type=float, help='Desfase en segundos del segundo vídeo')
parser.add_argument('-o', '--output', metavar='output_file', type=str, help='Ruta del archivo de salida')
parser.add_argument('-s', '--separated', action='store_true', help='Si se activa se guardará el primer vídeo desplazado en lugar de un vídeo compuesto over-under')

args = parser.parse_args()

video_files = args.input
offset = args.offset
output_file = args.output


print(f'Vídeos a sincronizar: {video_files}')
print(f'Desfase: {offset} segundos')
print(f'Archivo de salida: {output_file}')

# Función para ajustar la sincronización de los vídeos según el desfase
def adjust_video_offset(video_file_1, video_file_2, offset, output_file, separated=False):

    if not separated:
        # Usamos ffmpeg para ajustar el desfase y juntar los vídeos
        # El comando ffmpeg para desplazar el segundo vídeo
        command = [
            'ffmpeg',
            '-i', video_file_1,   # Primer vídeo
            '-itsoffset', str(offset),  # Desfase en segundos
            '-i', video_file_2,   # Segundo vídeo
            '-filter_complex', '[0:v][1:v]vstack=inputs=2[v]',  # Superponer los vídeos verticalmente
            '-map', '[v]',  # Mapear el vídeo compuesto
            '-y',            # Sobrescribir archivo de salida si ya existe        
            '-r', str(cv.VideoCapture(video_file_1).get(cv.CAP_PROP_FPS)), # Usar fps del primer vídeo
            '-vsync', '2',  # Evitar la duplicación de frames
            '-c:v', 'libx264',  # Codec de vídeo
            '-c:a', 'aac',      # Codec de audio
            '-y',               # Sobrescribir archivo de salida si ya existe
            output_file         # Archivo de salida
        ]
        subprocess.run(command)
    else:
        # Usamos ffmpeg para ajustar el desfase y guardar el primer vídeo desplazado
        command = [
            'ffmpeg',
            '-i', video_file_1,         # Primer vídeo
            '-itsoffset', str(offset),  # Desfase en segundos
            '-y',                       # Sobrescribir archivo de salida si ya existe        
            '-r', '60',                 # str(cv.VideoCapture(video_file_1).get(cv.CAP_PROP_FPS)), # Usar fps del primer vídeo
            '-vsync', '2',              # Evitar la duplicación de frames
            '-c:v', 'libx264',          # Codec de vídeo
            '-c:a', 'aac',              # Codec de audio
            '-y',                       # Sobrescribir archivo de salida si ya existe
            output_file                 # Archivo de salida
        ]
        subprocess.run(command)

# Crear el vídeo con los dos vídeos sincronizados
adjust_video_offset(video_files[0], video_files[1], offset, output_file)

print(f'El vídeo final ha sido creado en: {output_file}')
