import cv2
from ultralytics import YOLO

# Inicializa a webcam (0 = câmera padrão do sistema, troque para 1, 2... se tiver mais câmeras)
cap = cv2.VideoCapture(0)

# Define resolução (opcional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

# Carrega o modelo YOLOv8
model = YOLO("yolov8n.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Não conseguiu acessar a câmera.")
        break

    # Executa YOLO no frame
    results = model(frame)

    # Desenha as detecções
    annotated_frame = results[0].plot()

    # Calcula FPS
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time if inference_time > 0 else 0
    text = f'FPS: {fps:.1f}'

    # Escreve o FPS no canto superior direito
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Mostra a janela com detecção
    cv2.imshow("Camera", annotated_frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
