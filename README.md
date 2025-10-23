<div align="center">

# KLibras Model

### Script de Treinamento do Modelo de Reconhecimento de Libras

[![Python](https://img.shields.io/badge/Python-100%25-3776AB.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-00897B.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Treinamento de modelo de deep learning para reconhecimento de gestos de Libras usando GRU**

[Características](#características) • [Início Rápido](#início-rápido) • [Arquitetura](#arquitetura) • [Configuração](#configuração) • [Resultados](#resultados)

</div>

---

## Visão Geral

Este repositório contém o script completo para treinar o modelo de reconhecimento de gestos da Linguagem Brasileira de Sinais (Libras) usado no aplicativo KLibras. O modelo utiliza Redes Neurais Recorrentes (GRU) combinadas com MediaPipe para extração de features de pose corporal, mãos e face.


## Características

### Extração de Features
- **Pose Landmarker**: 33 pontos do corpo (x, y, z, visibility) = 132 features
- **Hand Landmarker**: 21 pontos por mão × 2 mãos (x, y, z) = 126 features
- **Face Landmarker**: 478 pontos faciais (x, y, z) = 1434 features
- **Total**: 1692 features por frame

### Arquitetura do Modelo
- **Entrada**: Sequências de 100 frames com 1692 features cada
- **Camadas GRU**: 3 camadas (64 → 128 → 64 unidades) com dropout
- **Camadas Dense**: 2 camadas totalmente conectadas (64 → 32 unidades)
- **Saída**: Classificação softmax para N classes de sinais

### Sistema de Treinamento
- **Train/Test Split**: 85% treino, 15% validação
- **Early Stopping**: Paciência de 20 épocas monitorando val_loss
- **Model Checkpoint**: Salva o melhor modelo baseado em val_accuracy
- **TensorBoard**: Logs detalhados para visualização do treinamento
- **Épocas**: Máximo de 150 com early stopping

### Visualizações Geradas
- **Histórico de Treinamento**: Loss e accuracy ao longo das épocas
- **Matriz de Confusão**: Normalizada e absoluta
- **Métricas por Classe**: Precision, Recall, F1-Score
- **Dashboard de Performance**: Resumo completo das métricas
- **Relatórios**: JSON e TXT com métricas detalhadas

---

## Início Rápido

### Pré-requisitos

- **Python 3.8+**
- **GPU com CUDA** (recomendado para treinamento rápido)
- **16GB+ RAM** (para processar vídeos)

### Instalação

#### 1. Clone o Repositório

```bash
git clone https://github.com/KLibras/model.git
cd model
```

#### 2. Instale as Dependências

```bash
pip install tensorflow opencv-python mediapipe scikit-learn matplotlib seaborn
```

Ou usando um arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Arquivos de Modelo Necessários

Baixe os modelos pré-treinados do MediaPipe e coloque na raiz do projeto:

- `pose_landmarker_lite.task` - Modelo de detecção de pose (~5.8 MB)
- `hand_landmarker.task` - Modelo de detecção de mãos (~7.8 MB)
- `face_landmarker.task` - Modelo de detecção facial (~3.8 MB)

**Links para download:**
- [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models)
- [Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models)
- [Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models)

### Estrutura de Dados

Organize seus vídeos de treinamento na seguinte estrutura:

```
model/
├── videos/
│   ├── obrigado/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   ├── null/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── [outras_classes]/
│       └── ...
├── train_model.py
├── pose_landmarker_lite.task
├── hand_landmarker.task
└── face_landmarker.task
```

### Configuração

Edite as constantes no início do arquivo `train_model.py`:

```python
# Caminho para os vídeos de treinamento
DATA_PATH = "videos"

# Classes/sinais a serem reconhecidos
ACTIONS = np.array(['obrigado', 'null'])

# Número de frames por sequência
SEQUENCE_LENGTH = 100

# Nome do arquivo do modelo a ser salvo
KERAS_MODEL_NAME = 'asl_action_recognizer.h5'
```

### Executar Treinamento

```bash
python train_model.py
```


## Estrutura do Projeto

```
model/
├── train_model.py              # Script principal de treinamento
├── pose_landmarker_lite.task   # Modelo MediaPipe de pose
├── hand_landmarker.task        # Modelo MediaPipe de mãos
├── face_landmarker.task        # Modelo MediaPipe de face
├── videos/                     # Diretório de vídeos de treino
├── training_plots/             # Gráficos gerados
│   ├── training_history_*.png
│   ├── confusion_matrix_*.png
│   ├── per_class_metrics_*.png
│   ├── performance_dashboard_*.png
│   ├── metrics_report_*.json
│   └── metrics_report_*.txt
├── Logs/                       # Logs do TensorBoard
└── asl_action_recognizer.h5    # Modelo treinado
```

---

### Pipeline de Processamento

```
Vídeo MP4
    ↓
Extração de Frames (OpenCV)
    ↓
MediaPipe Detection
    ├── Pose Landmarker (33 pontos × 4 valores)
    ├── Hand Landmarker (21 pontos × 3 valores × 2 mãos)
    └── Face Landmarker (478 pontos × 3 valores)
    ↓
Feature Vector (1692 features/frame)
    ↓
Normalização de Sequência (100 frames)
    ↓
Dataset (X, y)
    ↓
Modelo GRU
    ↓
Predição de Classe
```

### Arquitetura da Rede Neural

```
Input: (100, 1692)
    ↓
GRU(64, return_sequences=True)
    ↓
Dropout(0.2)
    ↓
GRU(128, return_sequences=True)
    ↓
Dropout(0.2)
    ↓
GRU(64, return_sequences=False)
    ↓
Dropout(0.2)
    ↓
Dense(64, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(32, activation='relu')
    ↓
Dense(num_classes, activation='softmax')
    ↓
Output: [probabilidades das classes]
```

## Detalhamento das Features

### Pose Landmarker (132 features)
33 pontos do corpo, cada um com 4 valores (x, y, z, visibility):
- Cabeça e pescoço
- Ombros, cotovelos, pulsos
- Quadril, joelhos, tornozelos
- Total: 33 × 4 = 132 features

### Hand Landmarker (126 features)
21 pontos por mão (esquerda e direita), cada um com 3 valores (x, y, z):
- Pulso
- Articulações dos dedos
- Total: 21 × 3 × 2 = 126 features

### Face Landmarker (1434 features)
478 pontos faciais, cada um com 3 valores (x, y, z):
- Contorno facial
- Olhos, sobrancelhas
- Nariz, boca
- Expressões faciais importantes para Libras
- Total: 478 × 3 = 1434 features

**Total Geral**: 132 + 126 + 1434 = **1692 features por frame**

---

## Configuração Avançada

### Ajustar Hiperparâmetros

Edite o arquivo `train_model.py`:

```python
# Mudar número de épocas
epochs = 150

# Ajustar early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,  # Ajuste aqui
    restore_best_weights=True,
    verbose=1
)

# Modificar arquitetura
model = Sequential([
    GRU(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, num_features)),  # Aumentar unidades
    Dropout(0.3),  # Ajustar dropout
    # ...
])
```

### Adicionar Novas Classes

1. Crie uma pasta com o nome da classe em `videos/`
2. Adicione vídeos .mp4 da classe
3. Atualize a constante `ACTIONS`:

```python
ACTIONS = np.array(['obrigado', 'null', 'qual_seu_nome', 'bom_dia', 'tudo_bem'])
```

### Visualizar Treinamento com TensorBoard

```bash
tensorboard --logdir=Logs
```

Acesse `http://localhost:6006` no navegador.

---

## Saídas Geradas

### Modelo Treinado
- **asl_action_recognizer.h5**: Modelo Keras salvo (~10-50 MB dependendo da arquitetura)

### Gráficos de Visualização
Todos salvos em `training_plots/` com timestamp:

1. **training_history_[timestamp].png**: 
   - Loss treino vs validação
   - Accuracy treino vs validação
   - Gaps de loss e accuracy

2. **confusion_matrix_[timestamp].png**:
   - Matriz de confusão absoluta
   - Matriz de confusão normalizada

3. **per_class_metrics_[timestamp].png**:
   - Precision, Recall, F1-Score por classe
   - Suporte (número de amostras)

4. **performance_dashboard_[timestamp].png**:
   - Dashboard completo com todas as métricas principais

### Relatórios
- **metrics_report_[timestamp].json**: Métricas em formato JSON
- **metrics_report_[timestamp].txt**: Relatório formatado em texto

---

## Exemplo de Uso

### Treinamento Básico

```bash
# 1. Preparar dados
mkdir -p videos/obrigado videos/null
# Adicionar vídeos .mp4 nas pastas

# 2. Baixar modelos MediaPipe
# Colocar .task files na raiz

# 3. Treinar
python train_model.py
```

### Treinamento com Múltiplas Classes

```python
# Editar train_model.py
ACTIONS = np.array(['obrigado', 'ola', 'tudo_bem', 'desculpa', 'null'])

# Criar estrutura de pastas
videos/
├── obrigado/
├── ola/
├── tudo_bem/
├── desculpa/
└── null/

# Executar
python train_model.py
```





## Métricas Esperadas

Com um dataset bem balanceado, espera-se:

- **Accuracy**: > 95%
- **Precision/Recall**: > 90% para cada classe
- **F1-Score**: > 92%
- **Loss (Validação)**: < 0.15

**Nota**: Métricas variam com qualidade e quantidade dos dados de treino.

---



### Para Treinamento Mais Rápido

1. **Use GPU**: CUDA com TensorFlow-GPU
2. **Reduza frames**: `SEQUENCE_LENGTH = 50`
3. **Use menos vídeos** para testes iniciais
4. **Reduza resolução** dos vídeos antes do processamento

### Para Melhor Acurácia

1. **Mais dados**: Mínimo 50-100 vídeos por classe
2. **Dados balanceados**: Mesmo número de vídeos por classe
3. **Vídeos diversificados**: Diferentes pessoas, ângulos, iluminação
4. **Data augmentation**: Implementar variações de velocidade, espelhamento



## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## Referências

- [MediaPipe Solutions](https://developers.google.com/mediapipe/solutions)
- [TensorFlow Keras](https://www.tensorflow.org/guide/keras)
- [GRU Networks](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)

---

## Suporte

- **Issues**: [GitHub Issues](https://github.com/KLibras/model/issues)
- **Discussões**: [GitHub Discussions](https://github.com/KLibras/model/discussions)

---

<div align="center">

**Script de treinamento do modelo KLibras - Democratizando o reconhecimento de Libras**

[Reportar Bug](https://github.com/KLibras/model/issues) • [Solicitar Funcionalidade](https://github.com/KLibras/model/issues)

</div>