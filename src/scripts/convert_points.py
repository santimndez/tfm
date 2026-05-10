import argparse
from convert import convert_points

parser = argparse.ArgumentParser(description='Genera código Python de inicialización de prompts de SAM2 a partir de un archivo de puntos seleccionado. Formato: columnas: frame x y isin.')
parser.add_argument('-i', '--input', type=str, help='Ruta al archivo de puntos seleccionado', required=True)
parser.add_argument('-o', '--output', type=str, help='Ruta al archivo de salida para guardar los puntos convertidos en código Python', default=None)

args = parser.parse_args()

convert_points(args.input, args.output)
print(f"Puntos convertidos de {args.input} y guardados en {args.output}")