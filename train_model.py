# pip install tensorflow opencv-python mediapipe scikit-learn matplotlib seaborn

# --- Bibliotecas e Módulos Importados ---
import cv2  # OpenCV para processamento de vídeo e imagem
import numpy as np  # NumPy para operações numéricas, especialmente com arrays
import os  # Módulo 'os' para interagir com o sistema operacional (ex: navegar por pastas)
from sklearn.model_selection import train_test_split  # Para dividir os dados em treino e teste
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from tensorflow.keras.utils import to_categorical  # Para converter rótulos em formato one-hot encoding
from tensorflow.keras.models import Sequential  # Modelo sequencial do Keras para construir a rede neural
from tensorflow.keras.layers import GRU, Dense, Dropout  # Tipos de camadas da rede (GRU para sequências, Densa para classificação)
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint  # Callbacks para melhorar o treinamento
import tensorflow as tf  # Biblioteca principal do TensorFlow
import mediapipe as mp  # Biblioteca do Google para detecção de corpo, mãos, etc.
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Suprime avisos de depreciação do MediaPipe para manter o output limpo
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Configuração de estilo para os gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# --- Seção 1: Constantes de Configuração ---
# É uma boa prática definir parâmetros importantes como constantes no início do script.

# Caminho para a pasta principal que contém as subpastas com os vídeos de cada ação.
DATA_PATH = "videos" # Mudar aqui
# Define as classes/ações que o modelo aprenderá a reconhecer.
ACTIONS = np.array(['obrigado', 'null'])
# Define o número fixo de frames que cada amostra de vídeo terá. Essencial para a entrada da rede neural.
SEQUENCE_LENGTH = 100
# Nome do arquivo do modelo Keras que será salvo.
KERAS_MODEL_NAME = 'asl_action_recognizer.h5'
# Diretório para salvar os gráficos
PLOTS_DIR = 'training_plots'
# Caminho para os arquivos de modelo do MediaPipe.
POSE_MODEL_PATH = 'pose_landmarker_lite.task'
HAND_MODEL_PATH = 'hand_landmarker.task'
FACE_MODEL_PATH = 'face_landmarker.task'

# Cria o diretório de gráficos se não existir
os.makedirs(PLOTS_DIR, exist_ok=True)

# Verifica se os modelos do MediaPipe existem, caso contrário, encerra o script.
if not os.path.exists(POSE_MODEL_PATH) or not os.path.exists(HAND_MODEL_PATH) or not os.path.exists(FACE_MODEL_PATH):
    print("="*80)
    print("ERRO: Por favor, baixe os modelos do MediaPipe (.task) e coloque-os neste diretório.")
    print(f"Pose Model: {os.path.exists(POSE_MODEL_PATH)}")
    print(f"Hand Model: {os.path.exists(HAND_MODEL_PATH)}")
    print(f"Face Model: {os.path.exists(FACE_MODEL_PATH)}")
    exit()

# Configuração dos detectores (Landmarkers) do MediaPipe.
base_options = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Define as opções para o detector de pose.
# O modo 'IMAGE' é usado para processar cada frame do vídeo individualmente de forma síncrona.
pose_options = PoseLandmarkerOptions(
    base_options=base_options(model_asset_path=POSE_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE)

# Define as opções para o detector de mãos, permitindo a detecção de até 2 mãos.
hand_options = HandLandmarkerOptions(
    base_options=base_options(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2)

# Define as opções para o detector de face.
face_options = FaceLandmarkerOptions(
    base_options=base_options(model_asset_path=FACE_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE)


# --- Seção 2: Função de Extração de Pontos-Chave ---

def extract_keypoints(pose_result, hand_result, face_result):
    """
    Extrai os pontos-chave (landmarks) do corpo, mãos e face a partir dos resultados do MediaPipe
    e os concatena em um único array NumPy.
    """
    # Extrai os 33 pontos da pose. Se nenhuma pose for detectada, cria um array de zeros.
    # Cada ponto tem 4 valores: x, y, z, e visibilidade. Total = 33 * 4 = 132 features.
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in pose_result.pose_landmarks[0]]).flatten() if pose_result.pose_landmarks else np.zeros(33 * 4)

    # Inicializa arrays de zeros para os 21 pontos de cada mão.
    # Cada ponto tem 3 valores: x, y, z. Total por mão = 21 * 3 = 63 features.
    lh, rh = np.zeros(21 * 3), np.zeros(21 * 3)
    
    # Se mãos forem detectadas, preenche os arrays correspondentes.
    if hand_result.hand_landmarks:
        for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
            # Verifica se é a mão esquerda ou direita.
            handedness = hand_result.handedness[i][0].category_name
            if handedness == "Left":
                lh = np.array([[res.x, res.y, res.z] for res in hand_landmarks]).flatten()
            elif handedness == "Right":
                rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks]).flatten()
    
    # Extrai os 478 pontos da face. Se nenhuma face for detectada, cria um array de zeros.
    # Cada ponto tem 3 valores: x, y, z. Total = 478 * 3 = 1434 features.
    face = np.array([[res.x, res.y, res.z] for res in face_result.face_landmarks[0]]).flatten() if face_result.face_landmarks else np.zeros(478 * 3)
                
    # Concatena os arrays de pose, mão esquerda, mão direita e face em um único vetor de características.
    # Total de features: 132 (pose) + 63 (mão esquerda) + 63 (mão direita) + 1434 (face) = 1692
    return np.concatenate([pose, lh, rh, face])


# --- Seção 3: Processamento de Vídeos e Carregamento de Dados ---

def process_videos_and_load_data():
    """
    Varre as pastas de vídeos, extrai os pontos-chave de cada frame, normaliza o tamanho
    das sequências e prepara os dados (X) e rótulos (y) para o treinamento.
    """
    print("Iniciando processamento de vídeos e carregamento de dados...")
    # Mapeia cada nome de ação para um número (ex: 'obrigado' -> 0, 'null' -> 1).
    label_map = {label: num for num, label in enumerate(ACTIONS)}
    sequences, labels = [], []

    # Utiliza 'with' para garantir que os recursos do MediaPipe sejam liberados corretamente.
    with PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
         HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
         FaceLandmarker.create_from_options(face_options) as face_landmarker:

        # Itera sobre cada ação definida (ex: 'obrigado', 'null').
        for action in ACTIONS:
            action_path = os.path.join(DATA_PATH, action)
            if not os.path.isdir(action_path):
                print(f"Aviso: Diretório não encontrado para a ação '{action}': {action_path}")
                continue

            print(f"Processando vídeos para a ação: '{action}'")
            video_count = 0
            # Itera sobre cada arquivo de vídeo na pasta da ação.
            for video_file in os.listdir(action_path):
                if not video_file.lower().endswith('.mp4'):
                    continue

                video_path = os.path.join(action_path, video_file)
                cap = cv2.VideoCapture(video_path)

                frame_landmarks = []
                # Loop para ler cada frame do vídeo.
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break  # Fim do vídeo.

                    # Converte o frame do formato BGR (OpenCV) para RGB e depois para o formato do MediaPipe.
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # Executa a detecção de pose, mãos e face no frame atual.
                    pose_result = pose_landmarker.detect(mp_image)
                    hand_result = hand_landmarker.detect(mp_image)
                    face_result = face_landmarker.detect(mp_image)

                    # Extrai os pontos-chave e os adiciona à lista de frames do vídeo.
                    keypoints = extract_keypoints(pose_result, hand_result, face_result)
                    frame_landmarks.append(keypoints)

                cap.release()

                # Após processar todos os frames, normaliza o comprimento da sequência.
                if len(frame_landmarks) > 0:
                    # Se o vídeo for mais longo que SEQUENCE_LENGTH, seleciona frames uniformemente.
                    if len(frame_landmarks) >= SEQUENCE_LENGTH:
                        indices = np.linspace(0, len(frame_landmarks) - 1, SEQUENCE_LENGTH, dtype=int)
                        sampled_landmarks = [frame_landmarks[i] for i in indices]
                    # Se o vídeo for mais curto, preenche com o último frame até atingir o comprimento.
                    else:
                        sampled_landmarks = frame_landmarks
                        padding = [frame_landmarks[-1]] * (SEQUENCE_LENGTH - len(frame_landmarks))
                        sampled_landmarks.extend(padding)
                    
                    # Adiciona a sequência normalizada e seu rótulo às listas principais.
                    sequences.append(sampled_landmarks)
                    labels.append(label_map[action])
                    video_count += 1
            
            print(f"  ✓ Processados {video_count} vídeos para '{action}'")

    # Converte as listas para arrays NumPy e os rótulos para o formato one-hot.
    return np.array(sequences), to_categorical(labels).astype(int)


# --- Seção 4: Funções de Visualização ---

def plot_training_history(history, timestamp):
    """
    Cria gráficos detalhados do histórico de treinamento.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📊 Histórico de Treinamento do Modelo ASL', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Loss (Treino vs Validação)
    axes[0, 0].plot(history.history['loss'], label='Treino', linewidth=2, marker='o', markersize=4)
    axes[0, 0].plot(history.history['val_loss'], label='Validação', linewidth=2, marker='s', markersize=4)
    axes[0, 0].set_title('Loss ao Longo das Épocas', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss (Categorical Crossentropy)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Accuracy (Treino vs Validação)
    axes[0, 1].plot(history.history['categorical_accuracy'], label='Treino', linewidth=2, marker='o', markersize=4)
    axes[0, 1].plot(history.history['val_categorical_accuracy'], label='Validação', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_title('Acurácia ao Longo das Épocas', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Acurácia')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Gráfico 3: Diferença entre Treino e Validação (Loss)
    loss_diff = np.array(history.history['loss']) - np.array(history.history['val_loss'])
    axes[1, 0].plot(loss_diff, linewidth=2, color='red', marker='o', markersize=4)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Gap de Loss (Treino - Validação)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Diferença de Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].fill_between(range(len(loss_diff)), loss_diff, alpha=0.3, color='red')
    
    # Gráfico 4: Learning Rate ao longo do tempo (se disponível)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], linewidth=2, color='green', marker='o', markersize=4)
        axes[1, 1].set_title('Taxa de Aprendizado', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Gráfico alternativo: Diferença de Acurácia
        acc_diff = np.array(history.history['categorical_accuracy']) - np.array(history.history['val_categorical_accuracy'])
        axes[1, 1].plot(acc_diff, linewidth=2, color='blue', marker='o', markersize=4)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Gap de Acurácia (Treino - Validação)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Diferença de Acurácia')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].fill_between(range(len(acc_diff)), acc_diff, alpha=0.3, color='blue')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'training_history_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de histórico salvo: training_history_{timestamp}.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, timestamp):
    """
    Cria uma matriz de confusão normalizada e não-normalizada.
    """
    # Converte one-hot encoding de volta para labels
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # Calcula a matriz de confusão
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('🎯 Matriz de Confusão', fontsize=16, fontweight='bold')
    
    # Matriz de confusão absoluta
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=ACTIONS, yticklabels=ACTIONS, 
                ax=axes[0], cbar_kws={'label': 'Contagem'})
    axes[0].set_title('Contagem Absoluta', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Classe Verdadeira')
    axes[0].set_xlabel('Classe Predita')
    
    # Matriz de confusão normalizada
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', 
                xticklabels=ACTIONS, yticklabels=ACTIONS, 
                ax=axes[1], cbar_kws={'label': 'Proporção'})
    axes[1].set_title('Normalizada (Proporção)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Classe Verdadeira')
    axes[1].set_xlabel('Classe Predita')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'confusion_matrix_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Matriz de confusão salva: confusion_matrix_{timestamp}.png")
    plt.close()


def plot_per_class_metrics(y_true, y_pred, timestamp):
    """
    Cria gráficos de métricas por classe (Precision, Recall, F1-Score).
    """
    # Converte one-hot encoding de volta para labels
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # Calcula métricas
    precision, recall, f1, support = precision_recall_fscore_support(y_true_labels, y_pred_labels)
    
    # Cria DataFrame para facilitar a visualização
    metrics_data = {
        'Classe': ACTIONS,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Suporte': support
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📈 Métricas por Classe', fontsize=16, fontweight='bold')
    
    x = np.arange(len(ACTIONS))
    width = 0.25
    
    # Gráfico 1: Precision, Recall, F1-Score lado a lado
    axes[0, 0].bar(x - width, precision, width, label='Precision', alpha=0.8)
    axes[0, 0].bar(x, recall, width, label='Recall', alpha=0.8)
    axes[0, 0].bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    axes[0, 0].set_xlabel('Classe')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Comparação de Métricas por Classe', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(ACTIONS, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1.1])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Gráfico 2: Suporte (número de amostras) por classe
    colors = plt.cm.viridis(np.linspace(0, 1, len(ACTIONS)))
    bars = axes[0, 1].bar(ACTIONS, support, color=colors, alpha=0.8)
    axes[0, 1].set_title('Suporte por Classe (Número de Amostras)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Número de Amostras')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    # Adiciona valores nas barras
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfico 3: Precision por classe (radar/polar se tiver muitas classes, senão barra horizontal)
    axes[1, 0].barh(ACTIONS, precision, color='skyblue', alpha=0.8)
    axes[1, 0].set_xlabel('Precision')
    axes[1, 0].set_title('Precision por Classe', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlim([0, 1])
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    # Adiciona valores
    for i, v in enumerate(precision):
        axes[1, 0].text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')
    
    # Gráfico 4: F1-Score por classe
    axes[1, 1].barh(ACTIONS, f1, color='lightcoral', alpha=0.8)
    axes[1, 1].set_xlabel('F1-Score')
    axes[1, 1].set_title('F1-Score por Classe', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    # Adiciona valores
    for i, v in enumerate(f1):
        axes[1, 1].text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'per_class_metrics_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Métricas por classe salvas: per_class_metrics_{timestamp}.png")
    plt.close()


def plot_model_performance_summary(history, y_test, y_pred, timestamp):
    """
    Cria um dashboard resumido com as principais métricas.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('🎯 Dashboard de Performance do Modelo ASL', fontsize=18, fontweight='bold')
    
    # 1. Loss final
    ax1 = fig.add_subplot(gs[0, 0])
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    ax1.bar(['Treino', 'Validação'], [final_train_loss, final_val_loss], color=['#3498db', '#e74c3c'], alpha=0.7)
    ax1.set_title('Loss Final', fontweight='bold')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate([final_train_loss, final_val_loss]):
        ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Acurácia final
    ax2 = fig.add_subplot(gs[0, 1])
    final_train_acc = history.history['categorical_accuracy'][-1]
    final_val_acc = history.history['val_categorical_accuracy'][-1]
    ax2.bar(['Treino', 'Validação'], [final_train_acc, final_val_acc], color=['#2ecc71', '#f39c12'], alpha=0.7)
    ax2.set_title('Acurácia Final', fontweight='bold')
    ax2.set_ylabel('Acurácia')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate([final_train_acc, final_val_acc]):
        ax2.text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Número de épocas treinadas
    ax3 = fig.add_subplot(gs[0, 2])
    total_epochs = len(history.history['loss'])
    ax3.text(0.5, 0.5, f'{total_epochs}', ha='center', va='center', fontsize=48, fontweight='bold', color='#9b59b6')
    ax3.text(0.5, 0.2, 'Épocas\nTreinadas', ha='center', va='center', fontsize=14, color='#34495e')
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.axis('off')
    
    # 4. Curva de Loss (grande)
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(history.history['loss'], label='Treino', linewidth=2.5, alpha=0.8)
    ax4.plot(history.history['val_loss'], label='Validação', linewidth=2.5, alpha=0.8)
    ax4.set_title('Evolução do Loss', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Época')
    ax4.set_ylabel('Loss')
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.fill_between(range(len(history.history['loss'])), history.history['loss'], alpha=0.2)
    ax4.fill_between(range(len(history.history['val_loss'])), history.history['val_loss'], alpha=0.2)
    
    # 5. Matriz de confusão mini
    ax5 = fig.add_subplot(gs[2, 0])
    y_true_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='RdYlGn', 
                xticklabels=ACTIONS, yticklabels=ACTIONS, ax=ax5, cbar=False, square=True)
    ax5.set_title('Matriz de Confusão', fontweight='bold')
    
    # 6. Métricas por classe
    ax6 = fig.add_subplot(gs[2, 1:])
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_labels, y_pred_labels)
    x = np.arange(len(ACTIONS))
    width = 0.25
    ax6.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax6.bar(x, recall, width, label='Recall', alpha=0.8)
    ax6.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    ax6.set_xlabel('Classe')
    ax6.set_ylabel('Score')
    ax6.set_title('Métricas por Classe', fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(ACTIONS)
    ax6.legend()
    ax6.set_ylim([0, 1.1])
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(os.path.join(PLOTS_DIR, f'performance_dashboard_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Dashboard de performance salvo: performance_dashboard_{timestamp}.png")
    plt.close()


def save_metrics_report(history, y_test, y_pred, timestamp):
    """
    Salva um relatório detalhado em formato JSON e texto.
    """
    y_true_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # Gera relatório de classificação
    report = classification_report(y_true_labels, y_pred_labels, target_names=ACTIONS, output_dict=True)
    
    # Adiciona informações do histórico de treinamento
    report['training_history'] = {
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_train_accuracy': float(history.history['categorical_accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_categorical_accuracy'][-1]),
        'total_epochs': len(history.history['loss']),
        'best_val_accuracy': float(max(history.history['val_categorical_accuracy'])),
        'best_val_accuracy_epoch': int(np.argmax(history.history['val_categorical_accuracy']) + 1)
    }
    
    # Salva como JSON
    json_path = os.path.join(PLOTS_DIR, f'metrics_report_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    # Salva como texto formatado
    txt_path = os.path.join(PLOTS_DIR, f'metrics_report_{timestamp}.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RELATÓRIO DE MÉTRICAS - MODELO ASL\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Modelo: {KERAS_MODEL_NAME}\n")
        f.write(f"Classes: {', '.join(ACTIONS)}\n\n")
        
        f.write("HISTÓRICO DE TREINAMENTO\n")
        f.write("-"*80 + "\n")
        f.write(f"Total de Épocas: {report['training_history']['total_epochs']}\n")
        f.write(f"Loss Final (Treino): {report['training_history']['final_train_loss']:.4f}\n")
        f.write(f"Loss Final (Validação): {report['training_history']['final_val_loss']:.4f}\n")
        f.write(f"Acurácia Final (Treino): {report['training_history']['final_train_accuracy']:.2%}\n")
        f.write(f"Acurácia Final (Validação): {report['training_history']['final_val_accuracy']:.2%}\n")
        f.write(f"Melhor Acurácia (Validação): {report['training_history']['best_val_accuracy']:.2%} (Época {report['training_history']['best_val_accuracy_epoch']})\n\n")
        
        f.write("RELATÓRIO DE CLASSIFICAÇÃO\n")
        f.write("-"*80 + "\n")
        f.write(classification_report(y_true_labels, y_pred_labels, target_names=ACTIONS))
        f.write("\n")
    
    print(f"✓ Relatório de métricas salvo: metrics_report_{timestamp}.json e .txt")


# --- Seção 5: Treinamento do Modelo ---

def train_model():
    """
    Carrega os dados, define a arquitetura da rede neural, compila, treina
    e salva o modelo finalizado em formato H5.
    """
    # Timestamp para identificar esta execução
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Carrega e processa os dados dos vídeos.
    X, y = process_videos_and_load_data()

    # Verifica se algum dado foi carregado antes de prosseguir.
    if X.shape[0] == 0:
        print("Erro: Nenhum dado foi carregado. Verifique o DATA_PATH e os arquivos de vídeo.")
        return

    # Divide os dados em 85% para treino e 15% para teste/validação.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    print(f"\n{'='*80}")
    print(f"Dados carregados e processados com sucesso.")
    print(f"Shape dos dados de treino: {X_train.shape}")
    print(f"Shape dos dados de teste: {X_test.shape}")
    print(f"Número total de amostras: {X.shape[0]}")
    print(f"Número de classes: {y.shape[1]}")
    print(f"{'='*80}\n")

    # Configura o TensorBoard para monitorar o treinamento.
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    
    # Early stopping para evitar overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    # Checkpoint para salvar o melhor modelo durante o treinamento
    checkpoint = ModelCheckpoint(
        KERAS_MODEL_NAME,
        monitor='val_categorical_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Define o número de features de entrada
    # 132 (pose) + 63 (mão esquerda) + 63 (mão direita) + 1434 (face) = 1692
    num_features = 1692

    # Define a arquitetura do modelo sequencial.
    model = Sequential([
        # Camadas GRU para aprender padrões temporais. 'return_sequences=True' é necessário
        # para passar a sequência completa para a próxima camada GRU.
        GRU(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, num_features)),
        Dropout(0.2),
        GRU(128, return_sequences=True),
        Dropout(0.2),
        # A última camada GRU não retorna a sequência, apenas o output final.
        GRU(64, return_sequences=False),
        Dropout(0.2),
        # Camadas densas para a classificação final.
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        # Camada de saída com ativação 'softmax' para problemas de classificação multiclasse.
        Dense(ACTIONS.shape[0], activation='softmax')
    ])

    # Compila o modelo, definindo o otimizador, a função de perda e as métricas.
    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    # Exibe um resumo da arquitetura do modelo.
    print("\n" + "="*80)
    print("ARQUITETURA DO MODELO")
    print("="*80)
    model.summary()
    print("="*80 + "\n")

    print("Iniciando treinamento do modelo...")
    print(f"Número de épocas: 150")
    print(f"Callbacks: TensorBoard, EarlyStopping, ModelCheckpoint\n")
    
    # Inicia o processo de treinamento.
    history = model.fit(
        X_train, y_train,
        epochs=150,
        callbacks=[tb_callback, early_stopping, checkpoint],
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    print("\n" + "="*80)
    print("Treinamento do modelo completo.")
    print("="*80)

    # Avalia o modelo nos dados de teste
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n📊 Resultados Finais:")
    print(f"   - Loss no conjunto de teste: {test_loss:.4f}")
    print(f"   - Acurácia no conjunto de teste: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Faz predições no conjunto de teste para gerar gráficos
    print("\n" + "="*80)
    print("GERANDO VISUALIZAÇÕES")
    print("="*80)
    y_pred = model.predict(X_test, verbose=0)
    
    # Gera todos os gráficos
    plot_training_history(history, timestamp)
    plot_confusion_matrix(y_test, y_pred, timestamp)
    plot_per_class_metrics(y_test, y_pred, timestamp)
    plot_model_performance_summary(history, y_test, y_pred, timestamp)
    save_metrics_report(history, y_test, y_pred, timestamp)

    # Salva o modelo treinado no formato H5 do Keras (se não foi salvo pelo checkpoint)
    model.save(KERAS_MODEL_NAME)
    print(f"\n💾 Modelo salvo como: {KERAS_MODEL_NAME}")
    print(f"   Tamanho do arquivo: {os.path.getsize(KERAS_MODEL_NAME) / (1024*1024):.2f} MB")


# --- Execução Principal ---
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 INICIANDO TREINAMENTO DE MODELO ASL")
    print("="*80)
    print(f"Ações a serem reconhecidas: {', '.join(ACTIONS)}")
    print(f"Comprimento da sequência: {SEQUENCE_LENGTH} frames")
    print(f"Detectores: Pose + Hands (2) + Face")
    print(f"Total de features por frame: 1692")
    print("="*80 + "\n")
    
    train_model()
    
    print("\n" + "="*80)
    print("✅ --- PROCESSO CONCLUÍDO COM SUCESSO --- ✅")
    print("="*80)
    print(f"\n📁 Arquivos gerados:")
    print(f"   - Modelo: {KERAS_MODEL_NAME}")
    print(f"   - Gráficos: ./{PLOTS_DIR}/ (vários arquivos PNG)")
    print(f"   - Relatórios: ./{PLOTS_DIR}/ (JSON e TXT)")
    print(f"   - Logs: ./Logs/ (visualize com TensorBoard)")
    print(f"\n💡 Para visualizar o treinamento no TensorBoard, execute:")
    print(f"   tensorboard --logdir=Logs")
    print("="*80 + "\n")