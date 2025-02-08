import cv2
import os

def extract_frame(input, nframe=0, output=None):
    # Cargar el video
    video = cv2.VideoCapture(input)

    if output == None:
        # Extract full path without extension 
        output = os.path.splitext(input)[0] + ".png"
    
    video.set(cv2.CAP_PROP_POS_FRAMES, nframe)
    success, frame = video.read()
    
    if success:
        cv2.imwrite(output, frame)  # Guardar el frame como imagen
    
    # Liberar el video
    video.release()
    
    return output if success else None


import argparse

# Define los parametros y llama a extract_frame
parser = argparse.ArgumentParser(description='Extraer un frame de un video')
parser.add_argument('-i', '--input', type=str, help='Ruta del video de entrada')
parser.add_argument('-f', '--frame', type=int, default=0, help='Número del frame a extraer')
parser.add_argument('-o', '--output', type=str, default=None, help='Ruta de la imagen de salida')

args = parser.parse_args()
output = extract_frame(args.input, args.frame, args.output)
print(args.frame)
if output:    
    print(f"Frame {args.frame} de {args.input} guardado como {output}")
else:
    print(f"Error al leer el video {args.input}")
