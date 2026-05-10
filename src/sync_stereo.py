import argparse
from utils import adjust_video_offset_ffmpeg

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

# Crear el vídeo con los dos vídeos sincronizados
adjust_video_offset_ffmpeg(video_files[0], video_files[1], offset, output_file)

print(f'El vídeo final ha sido creado en: {output_file}')
