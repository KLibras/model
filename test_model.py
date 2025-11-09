# pip install tensorflow opencv-python mediapipe scikit-learn matplotlib seaborn

# --- Bibliotecas e Módulos Importados ---
import cv2  # OpenCV para processamento de vídeo e imagem
import numpy as np  # NumPy para operações numéricas, especialmente com arrays
import os  # Módulo 'os' para interagir com o sistema operacional (ex: navegar por pastas)
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from tensorflow.keras.utils import to_categorical  # Para converter rótulos em formato one-hot encoding
from tensorflow.keras.models import load_model  # Importa a função para carregar um modelo treinado
import tensorflow as tf  # Biblioteca principal do TensorFlow
import mediapipe as mp  # Biblioteca do Google para detecção de corpo, mãos, face, etc.
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import multiprocessing as mp_process
from concurrent.futures import ProcessPoolExecutor, as_completed

# Suprime avisos de depreciação do MediaPipe para manter o output limpo
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Configuração de estilo para os gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# --- Seção 1: Constantes de Configuração ---
# É uma boa prática definir parâmetros importantes como constantes no início do script.

# Caminho para a pasta de dados de TESTE (ex: novos vídeos, novos signers)
TEST_DATA_PATH = "teste"
# Define as classes/ações que o modelo aprenderá a reconhecer.
ACTIONS = np.array(['obrigado', "tudo_bem", "bom_dia", "qual_seu_nome", 'null'])
# Define o número fixo de frames que cada amostra de vídeo terá. Essencial para a entrada da rede neural.
SEQUENCE_LENGTH = 100
# Nome do arquivo do modelo Keras que será CARREGADO.
KERAS_MODEL_NAME = 'klibras_model.h5'
# Diretório para salvar os gráficos e relatórios
RESULTS_DIR = 'test_results'
# Caminho para os arquivos de modelo do MediaPipe.
POSE_MODEL_PATH = 'pose_landmarker_lite.task'
HAND_MODEL_PATH = 'hand_landmarker.task'
FACE_MODEL_PATH = 'face_landmarker.task'
# Número de workers para processamento paralelo
NUM_WORKERS = 3  # Deixa 1 CPU livre para o sistema

# Cria o diretório de resultados se não existir
os.makedirs(RESULTS_DIR, exist_ok=True)

# Verifica se os modelos do MediaPipe e o modelo treinado existem
if not os.path.exists(KERAS_MODEL_NAME):
    print("="*80)
    print(f"ERRO: Modelo treinado '{KERAS_MODEL_NAME}' não encontrado.")
    print("Por favor, execute o script de treino primeiro.")
    exit()

if not os.path.exists(POSE_MODEL_PATH) or not os.path.exists(HAND_MODEL_PATH) or not os.path.exists(FACE_MODEL_PATH):
    print("="*80)
    print("ERRO: Por favor, baixe os modelos do MediaPipe (.task) e coloque-os neste diretório.")
    print(f"Pose Model: {os.path.exists(POSE_MODEL_PATH)}")
    print(f"Hand Model: {os.path.exists(HAND_MODEL_PATH)}")
    print(f"Face Model: {os.path.exists(FACE_MODEL_PATH)}")
    exit()


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


# --- Seção 3: Worker para Processamento Paralelo de Vídeos ---

def process_single_video(video_info):
    """
    Worker function para processar um único vídeo.
    Esta função será executada em paralelo por múltiplos processos.
    
    Args:
        video_info: Tupla contendo (action, video_path, label)
    
    Returns:
        Tupla (sequence, label) ou None se falhar
    """
    action, video_path, label = video_info
    
    try:
        # Configuração dos detectores do MediaPipe para este worker
        base_options = python.BaseOptions
        PoseLandmarker = vision.PoseLandmarker
        PoseLandmarkerOptions = vision.PoseLandmarkerOptions
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        FaceLandmarker = vision.FaceLandmarker
        FaceLandmarkerOptions = vision.FaceLandmarkerOptions
        VisionRunningMode = vision.RunningMode

        # Define as opções para cada detector
        pose_options = PoseLandmarkerOptions(
            base_options=base_options(model_asset_path=POSE_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE)

        hand_options = HandLandmarkerOptions(
            base_options=base_options(model_asset_path=HAND_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2)

        face_options = FaceLandmarkerOptions(
            base_options=base_options(model_asset_path=FACE_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE)

        # Cria os detectores para este worker
        with PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
             HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
             FaceLandmarker.create_from_options(face_options) as face_landmarker:

            cap = cv2.VideoCapture(video_path)
            frame_landmarks = []
            
            # Loop para ler cada frame do vídeo
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # Fim do vídeo

                # Converte o frame do formato BGR (OpenCV) para RGB e depois para o formato do MediaPipe
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Executa a detecção de pose, mãos e face no frame atual
                pose_result = pose_landmarker.detect(mp_image)
                hand_result = hand_landmarker.detect(mp_image)
                face_result = face_landmarker.detect(mp_image)

                # Extrai os pontos-chave e os adiciona à lista de frames do vídeo
                keypoints = extract_keypoints(pose_result, hand_result, face_result)
                frame_landmarks.append(keypoints)

            cap.release()

            # Normaliza o comprimento da sequência
            if len(frame_landmarks) > 0:
                # Se o vídeo for mais longo que SEQUENCE_LENGTH, seleciona frames uniformemente
                if len(frame_landmarks) >= SEQUENCE_LENGTH:
                    indices = np.linspace(0, len(frame_landmarks) - 1, SEQUENCE_LENGTH, dtype=int)
                    sampled_landmarks = [frame_landmarks[i] for i in indices]
                # Se o vídeo for mais curto, preenche com o último frame até atingir o comprimento
                else:
                    sampled_landmarks = frame_landmarks
                    padding = [frame_landmarks[-1]] * (SEQUENCE_LENGTH - len(frame_landmarks))
                    sampled_landmarks.extend(padding)
                
                return (np.array(sampled_landmarks), label)
            
    except Exception as e:
        print(f"Erro ao processar {video_path}: {str(e)}")
        return None
    
    return None


def process_test_data(data_path):
    """
    Varre as pastas de vídeos, extrai os pontos-chave de cada frame usando workers paralelos,
    normaliza o tamanho das sequências e prepara os dados (X) e rótulos (y) para o teste.
    """
    print(f"Iniciando processamento de vídeos do diretório: {data_path}")
    print(f"Usando {NUM_WORKERS} workers para processamento paralelo")
    
    # Mapeia cada nome de ação para um número (ex: 'obrigado' -> 0, 'null' -> 1)
    label_map = {label: num for num, label in enumerate(ACTIONS)}
    
    # Coleta informações de todos os vídeos a serem processados
    video_jobs = []
    for action in ACTIONS:
        action_path = os.path.join(data_path, action)
        if not os.path.isdir(action_path):
            print(f"Aviso: Diretório não encontrado para a ação '{action}': {action_path}")
            continue

        for video_file in os.listdir(action_path):
            if not video_file.lower().endswith('.mp4'):
                continue
            
            video_path = os.path.join(action_path, video_file)
            video_jobs.append((action, video_path, label_map[action]))
    
    print(f"Total de vídeos a processar em '{data_path}': {len(video_jobs)}")
    
    sequences, labels = [], []
    processed_count = 0
    failed_count = 0
    
    # Processa os vídeos em paralelo usando ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submete todas as tarefas
        futures = {executor.submit(process_single_video, job): job for job in video_jobs}
        
        # Coleta os resultados conforme ficam prontos
        for future in as_completed(futures):
            job = futures[future]
            action, video_path, label = job
            
            try:
                result = future.result()
                if result is not None:
                    sequence, label = result
                    sequences.append(sequence)
                    labels.append(label)
                    processed_count += 1
                    
                    # Mostra progresso a cada 10 vídeos
                    if processed_count % 10 == 0:
                        print(f"  Progresso ({data_path}): {processed_count}/{len(video_jobs)} vídeos processados")
                else:
                    failed_count += 1
                    
            except Exception as e:
                print(f"Erro ao processar {video_path}: {str(e)}")
                failed_count += 1
    
    print(f"\n✓ Processamento de '{data_path}' concluído!")
    print(f"  - Vídeos processados com sucesso: {processed_count}")
    print(f"  - Vídeos com falha: {failed_count}")
    
    # Exibe estatísticas por ação
    print(f"\n📊 Distribuição de amostras ({data_path}):")
    for action_idx, action in enumerate(ACTIONS):
        count = sum(1 for l in labels if l == action_idx)
        print(f"  - {action}: {count} vídeos")
    
    # Converte as listas para arrays NumPy e os rótulos para o formato one-hot
    return np.array(sequences), to_categorical(labels).astype(int)


# --- Seção 4: Funções de Visualização ---

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
    fig.suptitle('🎯 Matriz de Confusão (Dados de Teste)', fontsize=16, fontweight='bold')
    
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
    plt.savefig(os.path.join(RESULTS_DIR, f'test_confusion_matrix_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Matriz de confusão salva: test_confusion_matrix_{timestamp}.png")
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
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📈 Métricas por Classe (Dados de Teste)', fontsize=16, fontweight='bold')
    
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
    
    # Gráfico 3: Precision por classe
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
    plt.savefig(os.path.join(RESULTS_DIR, f'test_per_class_metrics_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Métricas por classe salvas: test_per_class_metrics_{timestamp}.png")
    plt.close()


def plot_evaluation_dashboard(y_test, y_pred, test_loss, test_accuracy, timestamp):
    """
    Cria um dashboard resumido com as principais métricas de teste.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    fig.suptitle('🎯 Dashboard de Performance (Dados de Teste)', fontsize=18, fontweight='bold')
    
    # 1. Acurácia Final
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.6, f'{test_accuracy:.2%}', ha='center', va='center', fontsize=48, fontweight='bold', color='#2ecc71')
    ax1.text(0.5, 0.3, 'Acurácia Geral', ha='center', va='center', fontsize=18, color='#34495e')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.axis('off')

    # 2. Loss Final
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.6, f'{test_loss:.4f}', ha='center', va='center', fontsize=48, fontweight='bold', color='#e74c3c')
    ax2.text(0.5, 0.3, 'Loss Geral', ha='center', va='center', fontsize=18, color='#34495e')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.axis('off')
    
    # 3. Matriz de confusão
    ax3 = fig.add_subplot(gs[1, 0])
    y_true_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='RdYlGn', 
                xticklabels=ACTIONS, yticklabels=ACTIONS, ax=ax3, cbar=False, square=True)
    ax3.set_title('Matriz de Confusão (Normalizada)', fontweight='bold')
    
    # 4. F1-Score por Classe
    ax4 = fig.add_subplot(gs[1, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_labels, y_pred_labels)
    ax4.barh(ACTIONS, f1, color='lightcoral', alpha=0.8)
    ax4.set_xlabel('F1-Score')
    ax4.set_title('F1-Score por Classe', fontweight='bold')
    ax4.set_xlim([0, 1])
    ax4.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(f1):
        ax4.text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'test_performance_dashboard_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Dashboard de performance salvo: test_performance_dashboard_{timestamp}.png")
    plt.close()


def save_evaluation_report(y_test, y_pred, test_loss, test_accuracy, timestamp):
    """
    Salva um relatório de avaliação detalhado em formato JSON e texto.
    """
    y_true_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # Gera relatório de classificação
    report = classification_report(y_true_labels, y_pred_labels, target_names=ACTIONS, output_dict=True)
    
    # Adiciona informações da avaliação
    report['evaluation_metrics'] = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
    }
    
    # Salva como JSON
    json_path = os.path.join(RESULTS_DIR, f'test_metrics_report_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    # Salva como texto formatado
    txt_path = os.path.join(RESULTS_DIR, f'test_metrics_report_{timestamp}.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RELATÓRIO DE AVALIAÇÃO - MODELO LIBRAS (Dados de Teste)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Modelo Testado: {KERAS_MODEL_NAME}\n")
        f.write(f"Classes: {', '.join(ACTIONS)}\n")
        
        f.write("MÉTRICAS GERAIS\n")
        f.write("-"*80 + "\n")
        f.write(f"Loss no conjunto de teste: {test_loss:.4f}\n")
        f.write(f"Acurácia no conjunto de teste: {test_accuracy:.2%}\n\n")
        
        f.write("RELATÓRIO DE CLASSIFICAÇÃO\n")
        f.write("-"*80 + "\n")
        f.write(classification_report(y_true_labels, y_pred_labels, target_names=ACTIONS))
        f.write("\n")
    
    print(f"✓ Relatório de métricas salvo: test_metrics_report_{timestamp}.json e .txt")


# --- Seção 5: Avaliação do Modelo ---

def evaluate_model():
    """
    Carrega os dados de teste, carrega o modelo treinado,
    avalia a performance e gera todos os relatórios visuais.
    """
    # Timestamp para identificar esta execução
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Carrega e processa os dados dos vídeos de teste
    print("\n" + "="*80)
    print(f"CARREGANDO DADOS DE TESTE DE '{TEST_DATA_PATH}'")
    X_test, y_test = process_test_data(TEST_DATA_PATH)
    
    # Verifica se algum dado foi carregado antes de prosseguir
    if X_test.shape[0] == 0:
        print(f"Erro: Nenhum dado foi carregado de '{TEST_DATA_PATH}'. Verifique o diretório.")
        return

    print(f"\n{'='*80}")
    print(f"Dados de teste carregados e processados com sucesso.")
    print(f"Shape dos dados de teste: {X_test.shape}")
    print(f"Total de amostras de teste: {X_test.shape[0]}")
    print(f"Número de classes: {y_test.shape[1]}")
    print(f"{'='*80}\n")
    
    # Carrega o modelo H5 treinado
    print(f"Carregando modelo treinado: {KERAS_MODEL_NAME}...")
    try:
        model = load_model(KERAS_MODEL_NAME)
        print("✓ Modelo carregado com sucesso.")
        model.summary()
    except Exception as e:
        print(f"ERRO AO CARREGAR O MODELO: {str(e)}")
        return
    print("="*80 + "\n")

    print("Iniciando avaliação do modelo nos dados de teste...")
    
    # Avalia o modelo nos dados de teste
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    
    print("\n" + "="*80)
    print("Avaliação completa.")
    print("="*80)
    
    print(f"\n📊 Resultados Finais (Conjunto de Teste '{TEST_DATA_PATH}'):")
    print(f"   - Loss no conjunto de teste: {test_loss:.4f}")
    print(f"   - Acurácia no conjunto de teste: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Faz predições no conjunto de teste para gerar gráficos
    print("\n" + "="*80)
    print("GERANDO VISUALIZAÇÕES DE TESTE")
    print("="*80)
    y_pred = model.predict(X_test, verbose=0)
    
    # Gera todos os gráficos
    plot_confusion_matrix(y_test, y_pred, timestamp)
    plot_per_class_metrics(y_test, y_pred, timestamp)
    plot_evaluation_dashboard(y_test, y_pred, test_loss, test_accuracy, timestamp)
    save_evaluation_report(y_test, y_pred, test_loss, test_accuracy, timestamp)

    print(f"\n✓ Avaliação e geração de relatórios concluídas.")


# --- Execução Principal ---
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 INICIANDO AVALIAÇÃO DE MODELO LIBRAS EM DADOS DE TESTE")
    print("="*80)
    print(f"Ações a serem reconhecidas: {', '.join(ACTIONS)}")
    print(f"Comprimento da sequência: {SEQUENCE_LENGTH} frames")
    print(f"Detectores: Pose + Hands (2) + Face")
    print(f"Total de features por frame: 1692")
    print(f"Número de workers: {NUM_WORKERS}")
    print(f"Dados de teste: ./{TEST_DATA_PATH}")
    print(f"Modelo: {KERAS_MODEL_NAME}")
    print("="*80 + "\n")
    
    evaluate_model()
    
    print("\n" + "="*80)
    print("✅ --- PROCESSO DE AVALIAÇÃO CONCLUÍDO --- ✅")
    print("="*80)
    print(f"\n📁 Arquivos gerados:")
    print(f"   - Gráficos: ./{RESULTS_DIR}/ (vários arquivos PNG)")
    print(f"   - Relatórios: ./{RESULTS_DIR}/ (JSON e TXT)")
    print("="*80 + "\n")