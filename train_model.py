# pip install tensorflow opencv-python mediapipe scikit-learn matplotlib seaborn

import cv2
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# --- Constantes ---
TRAIN_DATA_PATH = "sinais"
TEST_DATA_PATH = "teste"
ACTIONS = np.array(['obrigado', "tudo_bem", "bom_dia", "qual_seu_nome", 'null'])
SEQUENCE_LENGTH = 100
KERAS_MODEL_NAME = 'klibras_model.h5'
RESULTS_DIR = 'training_results'
POSE_MODEL_PATH = 'pose_landmarker_lite.task'
HAND_MODEL_PATH = 'hand_landmarker.task'
FACE_MODEL_PATH = 'face_landmarker.task'
NUM_WORKERS = 6

os.makedirs(RESULTS_DIR, exist_ok=True)

if not os.path.exists(POSE_MODEL_PATH) or not os.path.exists(HAND_MODEL_PATH) or not os.path.exists(FACE_MODEL_PATH):
    print("="*80)
    print("ERRO: Modelos do MediaPipe não encontrados.")
    exit()


def extract_keypoints(pose_result, hand_result, face_result):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in pose_result.pose_landmarks[0]]).flatten() if pose_result.pose_landmarks else np.zeros(33 * 4)
    lh, rh = np.zeros(21 * 3), np.zeros(21 * 3)
    
    if hand_result.hand_landmarks:
        for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[i][0].category_name
            if handedness == "Left":
                lh = np.array([[res.x, res.y, res.z] for res in hand_landmarks]).flatten()
            elif handedness == "Right":
                rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks]).flatten()
    
    face = np.array([[res.x, res.y, res.z] for res in face_result.face_landmarks[0]]).flatten() if face_result.face_landmarks else np.zeros(478 * 3)
    return np.concatenate([pose, lh, rh, face])


def process_single_video(video_info):
    action, video_path, label = video_info
    
    try:
        base_options = python.BaseOptions
        PoseLandmarker = vision.PoseLandmarker
        PoseLandmarkerOptions = vision.PoseLandmarkerOptions
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        FaceLandmarker = vision.FaceLandmarker
        FaceLandmarkerOptions = vision.FaceLandmarkerOptions
        VisionRunningMode = vision.RunningMode

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

        with PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
             HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
             FaceLandmarker.create_from_options(face_options) as face_landmarker:

            cap = cv2.VideoCapture(video_path)
            frame_landmarks = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pose_result = pose_landmarker.detect(mp_image)
                hand_result = hand_landmarker.detect(mp_image)
                face_result = face_landmarker.detect(mp_image)
                keypoints = extract_keypoints(pose_result, hand_result, face_result)
                frame_landmarks.append(keypoints)

            cap.release()

            if len(frame_landmarks) > 0:
                if len(frame_landmarks) >= SEQUENCE_LENGTH:
                    indices = np.linspace(0, len(frame_landmarks) - 1, SEQUENCE_LENGTH, dtype=int)
                    sampled_landmarks = [frame_landmarks[i] for i in indices]
                else:
                    sampled_landmarks = frame_landmarks
                    padding = [frame_landmarks[-1]] * (SEQUENCE_LENGTH - len(frame_landmarks))
                    sampled_landmarks.extend(padding)
                
                return (np.array(sampled_landmarks), label)
            
    except Exception as e:
        print(f"Erro ao processar {video_path}: {str(e)}")
        return None
    
    return None


def augment_single_sample(seq):
    """Aplica UMA augmentação aleatória em uma amostra"""
    aug_type = np.random.randint(0, 8)
    
    if aug_type == 0:
        # Original
        return seq
    elif aug_type == 1:
        # Horizontal flip
        augmented = seq.copy()
        for i in range(len(augmented)):
            for j in range(0, len(augmented[i]), 3):
                augmented[i][j] = 1.0 - augmented[i][j]
        return augmented
    elif aug_type == 2:
        # Rotation
        angle = np.random.uniform(-10, 10)
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        augmented = seq.copy()
        for i in range(len(augmented)):
            for j in range(0, len(augmented[i])-1, 3):
                x, y = augmented[i][j], augmented[i][j+1]
                x_c, y_c = x - 0.5, y - 0.5
                augmented[i][j] = x_c * cos_a - y_c * sin_a + 0.5
                augmented[i][j+1] = x_c * sin_a + y_c * cos_a + 0.5
        return augmented
    elif aug_type == 3:
        # Zoom
        zoom = np.random.uniform(0.9, 1.1)
        augmented = seq.copy()
        for i in range(len(augmented)):
            for j in range(0, len(augmented[i])-1, 3):
                x, y = augmented[i][j], augmented[i][j+1]
                augmented[i][j] = (x - 0.5) * zoom + 0.5
                augmented[i][j+1] = (y - 0.5) * zoom + 0.5
        return augmented
    elif aug_type == 4:
        # Shift
        shift_x = np.random.uniform(-0.05, 0.05)
        shift_y = np.random.uniform(-0.05, 0.05)
        augmented = seq.copy()
        for i in range(len(augmented)):
            for j in range(0, len(augmented[i])-1, 3):
                augmented[i][j] = np.clip(augmented[i][j] + shift_x, 0, 1)
                augmented[i][j+1] = np.clip(augmented[i][j+1] + shift_y, 0, 1)
        return augmented
    elif aug_type == 5:
        # Noise
        noise_level = np.random.uniform(0.005, 0.02)
        augmented = seq + np.random.normal(0, noise_level, seq.shape)
        return np.clip(augmented, 0, 1)
    elif aug_type == 6:
        # Reverse
        return seq[::-1].copy()
    elif aug_type == 7:
        # Speed
        speed = np.random.uniform(0.8, 1.2)
        new_len = int(SEQUENCE_LENGTH * speed)
        if new_len > 0:
            indices = np.linspace(0, len(seq) - 1, new_len, dtype=int)
            augmented = np.array([seq[i] for i in indices])
            final_indices = np.linspace(0, len(augmented) - 1, SEQUENCE_LENGTH, dtype=int)
            return np.array([augmented[i] for i in final_indices])
        return seq


def simple_augment_data(sequences, labels, multiplier=8):
    """
    Augmenta dados de forma simples - cria cópias com augmentação aplicada
    """
    print(f"\n{'='*80}")
    print("🔥 APLICANDO AUGMENTAÇÃO DE DADOS (MÉTODO SIMPLES)")
    print(f"{'='*80}")
    print(f"Dataset original: {len(sequences)} amostras")
    print(f"Multiplicador: {multiplier}x")
    
    augmented_seqs = []
    augmented_labels = []
    
    # Adiciona originais
    for seq, label in zip(sequences, labels):
        augmented_seqs.append(seq)
        augmented_labels.append(label)
    
    # Adiciona versões augmentadas
    for _ in range(multiplier - 1):
        for seq, label in zip(sequences, labels):
            aug_seq = augment_single_sample(seq)
            augmented_seqs.append(aug_seq)
            augmented_labels.append(label)
    
    print(f"✓ Augmentação concluída!")
    print(f"  Dataset aumentado: {len(augmented_seqs)} amostras")
    print(f"{'='*80}\n")
    
    return np.array(augmented_seqs), np.array(augmented_labels)


def load_data_from_path(data_path):
    print(f"Processando vídeos de: {data_path}")
    print(f"Workers: {NUM_WORKERS}")
    
    label_map = {label: num for num, label in enumerate(ACTIONS)}
    video_jobs = []
    
    for action in ACTIONS:
        action_path = os.path.join(data_path, action)
        if not os.path.isdir(action_path):
            continue
        for video_file in os.listdir(action_path):
            if video_file.lower().endswith('.mp4'):
                video_jobs.append((action, os.path.join(action_path, video_file), label_map[action]))
    
    print(f"Total de vídeos: {len(video_jobs)}")
    
    sequences, labels = [], []
    processed = 0
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_video, job): job for job in video_jobs}
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                sequences.append(result[0])
                labels.append(result[1])
                processed += 1
                if processed % 10 == 0:
                    print(f"  Progresso: {processed}/{len(video_jobs)}")
    
    print(f"\n✓ Processados: {processed} vídeos")
    
    for action_idx, action in enumerate(ACTIONS):
        count = sum(1 for l in labels if l == action_idx)
        print(f"  - {action}: {count}")
    
    return np.array(sequences), to_categorical(labels).astype(int)


def plot_training_history(history, timestamp):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📊 Histórico de Treinamento', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(history.history['loss'], label='Treino', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validação', linewidth=2)
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history.history['categorical_accuracy'], label='Treino', linewidth=2)
    axes[0, 1].plot(history.history['val_categorical_accuracy'], label='Validação', linewidth=2)
    axes[0, 1].set_title('Acurácia')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'training_{timestamp}.png'), dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, timestamp):
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ACTIONS, yticklabels=ACTIONS, ax=ax)
    ax.set_title('Matriz de Confusão')
    ax.set_ylabel('Verdadeiro')
    ax.set_xlabel('Predito')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'confusion_{timestamp}.png'), dpi=300)
    plt.close()


def save_metrics(history, y_test, y_pred, timestamp):
    y_true = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    report = classification_report(y_true, y_pred_labels, target_names=ACTIONS, output_dict=True)
    report['history'] = {
        'epochs': len(history.history['loss']),
        'final_train_acc': float(history.history['categorical_accuracy'][-1]),
        'final_val_acc': float(history.history['val_categorical_accuracy'][-1])
    }
    
    with open(os.path.join(RESULTS_DIR, f'metrics_{timestamp}.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"✓ Métricas salvas")


def train_model():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print("CARREGANDO DADOS")
    print("="*80)
    
    X_train, y_train = load_data_from_path(TRAIN_DATA_PATH)
    X_test, y_test = load_data_from_path(TEST_DATA_PATH)
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("ERRO: Nenhum dado carregado")
        return
    
    # Augmenta apenas dados de treino (4x)
    X_train, y_train = simple_augment_data(X_train, y_train, multiplier=4)
    
    print(f"\nDados finais:")
    print(f"  Treino: {X_train.shape}")
    print(f"  Teste: {X_test.shape}\n")
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(KERAS_MODEL_NAME, monitor='val_categorical_accuracy', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1)
    
    # Modelo
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)), 
                     input_shape=(SEQUENCE_LENGTH, 1692)),
        BatchNormalization(),
        Dropout(0.4),
        
        Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))),
        BatchNormalization(),
        Dropout(0.4),
        
        Bidirectional(LSTM(128, return_sequences=False, kernel_regularizer=regularizers.l2(0.001))),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.4),
        
        Dense(len(ACTIONS), activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    print("\n" + "="*80)
    print("MODELO")
    print("="*80)
    model.summary()
    print("="*80 + "\n")
    
    print("Iniciando treinamento...\n")
    
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1
    )
    
    print("\n" + "="*80)
    print("TREINAMENTO COMPLETO")
    print("="*80)
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nResultados no teste:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Acurácia: {acc:.4f} ({acc*100:.2f}%)")
    
    y_pred = model.predict(X_test, verbose=0)
    
    plot_training_history(history, timestamp)
    plot_confusion_matrix(y_test, y_pred, timestamp)
    save_metrics(history, y_test, y_pred, timestamp)
    
    model.save(KERAS_MODEL_NAME)
    print(f"\n💾 Modelo salvo: {KERAS_MODEL_NAME}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 TREINAMENTO COM AUGMENTAÇÃO SIMPLES (SEM BUGS)")
    print("="*80)
    print(f"Classes: {', '.join(ACTIONS)}")
    print(f"Workers: {NUM_WORKERS}")
    print("="*80 + "\n")
    
    train_model()
    
    print("\n" + "="*80)
    print("✅ CONCLUÍDO")
    print("="*80)