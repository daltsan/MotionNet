# Relatório de Configuração e Integração: MotionNet & CARMEN

Este documento descreve detalhadamente a configuração do ambiente, o fluxo de controle de versão (Git) e as modificações e criações de código realizadas para integrar o modelo de percepção e predição de movimento **MotionNet** com a base de dados/logs do ecossistema **CARMEN (LCAD/UFES)**.

---

## 1. Controle de Versão (Repositório)

* **Fork no repositório?** Não foi feito um fork. O repositório original do MotionNet ([merlresearch/MotionNet](https://github.com/merlresearch/MotionNet.git)) foi clonado diretamente para a máquina.
* **Estado do Git**: O repositório local está na branch `main` e aponta diretamente para o repositório upstream original como `origin`.

---

## 2. Configuração do Ambiente (Virtual Environment)

Para rodar o projeto e evitar conflitos de dependências, o ambiente virtual Python (`venv`) deve ser criado e configurado localmente.

> [!NOTE]
> Pastas de ambiente virtual (`venv/`) **não** são enviadas para o repositório Git. Elas contêm binários específicos do sistema operacional e caminhos absolutos locais que não funcionariam em outras máquinas. Em vez disso, as dependências necessárias estão salvas no arquivo `requirements.txt`.

### Como recriar o ambiente em qualquer máquina:

1. **Criar o ambiente virtual** (no diretório raiz):
   ```bash
   python3 -m venv venv
   ```
2. **Ativar o ambiente**:
   ```bash
   source venv/bin/activate
   ```
3. **Instalar todas as dependências**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Configuração de Variáveis de Ambiente**:
   Para expor o pacote `MotionNet` e o kit de desenvolvimento do nuScenes, a variável `PYTHONPATH` precisa ser configurada no terminal:
   ```bash
   export PYTHONPATH=$(pwd)/MotionNet:$(pwd)/MotionNet/nuscenes-devkit/python-sdk
   ```

### Principais Dependências Instaladas
O ambiente virtual foi populado com as bibliotecas necessárias para deep learning (PyTorch), manipulação geométrica de nuvens de pontos e visualização:
* **PyTorch (2.12.0)**: Biblioteca de deep learning para rodar o modelo e inferências.
* **NumPy (2.4.6) & SciPy (1.17.1)**: Para processamento matemático e interpolação de odometria.
* **nuscenes-devkit (1.2.0)**: SDK oficial da nuScenes, utilizado para gerenciar estruturas de representação espacial tridimensional.
* **Shapely & PyQuaternion**: Para cálculos geométricos de caixas delimitadoras (bounding boxes) e rotações em 3D.
* **Matplotlib & ImageIO**: Ferramentas usadas para desenhar e salvar os gráficos das projeções de LiDAR e exportar os GIFs de resultados.
* **Numba**: Para aceleração JIT (Just-In-Time) na voxelização das nuvens de pontos.

---

## 3. Alterações na Base de Código Original (MotionNet)

A base de código original continha algumas incompatibilidades com versões mais novas de Python e NumPy, além de algumas dependências rígidas com o dataset nuScenes completo. Foram realizadas as seguintes modificações em arquivos existentes do repositório:

1. **Correções de Compatibilidade com NumPy Novo**:
   * Arquivos modificados:
     * data/data_utils.py
     * data/gen_data.py
     * data/nuscenes_dataloader.py
     * plots.py
     * nuscenes-devkit/python-sdk/nuscenes/utils/map_mask.py
   * **Modificação**: Os aliases `np.bool` e `np.int`, que foram removidos a partir do NumPy 1.24+, foram substituídos pelos tipos nativos do Python `bool` e `int` para evitar exceções de `AttributeError` durante a execução.

2. **Adaptação para nuScenes-mini**:
   * Arquivo modificado: data/gen_data.py
   * **Modificação**: O código original esperava ler o mapeamento completo do dataset nuScenes em arquivo de splits (`data/split.npy`). Foi adicionada uma regra especial para carregar diretamente as cenas da versão simplificada `nuScenes-mini` (`scene_103`, `scene_1100`, etc.) caso o split especificado inclua a palavra `"mini"`.

---

## 4. Scripts Criados para Integração com o CARMEN

Foram criados três novos scripts para implementar a ponte entre os logs brutos gerados pelo veículo/simulador do CARMEN e o formato de entrada que o MotionNet consome.

### A. `data/carmen_to_motionnet.py`
Este script faz o pré-processamento pesado dos logs brutos do CARMEN:
* **Leitura da Nuvem de Pontos Binária**: Abre os arquivos `.pointcloud` gerados pelo Velodyne de 32 canais e lê as mensagens compactadas em formato binário (usando a biblioteca `struct`). Converte as leituras de distância e intensidade para coordenadas espaciais retangulares (X, Y, Z).
* **Alinhamento de Eixos**: Adapta a orientação dos eixos de coordenadas cartesianas do CARMEN para coincidir com o padrão do nuScenes (onde X aponta para a direita, Y para a frente e Z para cima).
* **Integração de Odometria**: Lê o arquivo de log para encontrar mensagens `ROBOTVELOCITY_ACK` (velocidade linear e angular com carimbo de tempo). Faz a integração numérica de Euler para gerar uma trajetória contínua da pose da plataforma móvel (posição X, Y e ângulo $\theta$).
* **Transformação Temporal (Spatiotemporal Alignment)**: Para cada frame a ser gerado, o script lê a nuvem de pontos atual e as 4 nuvens passadas anteriores (total de 5 sweeps consecutivas). Usando a odometria calculada, ele rotaciona e translada os pontos do passado de forma a alinhá-los ao referencial do veículo no instante atual.
* **Voxelização**: As nuvens são consolidadas em uma grade de voxels ocupacionais em Bird's Eye View (BEV) de dimensão `256 x 256 x 13` (cobrindo uma área de $64\text{m} \times 64\text{m}$ em torno do veículo com resolução espacial de $0.25\text{m}$ por célula). Os resultados são salvos em arquivos compactados `.npy`.

### B. `test_carmen.py`
Este script carrega o modelo pré-treinado e gera as predições a partir dos dados convertidos:
* **Execução do Modelo**: Carrega os pesos do modelo (`model.pth`) em PyTorch e realiza a inferência em batch a partir das grades de voxels `.npy` geradas.
* **Predição Multitarefa**: O MotionNet fornece três saídas principais:
  1. Campo vetorial de deslocamento dinâmico futuro (`disp_pred`).
  2. Predição semântica da categoria dos objetos ocupantes (`cat_pred`).
  3. Classificação se o estado de movimento é dinâmico ou estático (`motion_pred`).
* **Mascaramento e Refinamento**: Utiliza o estado de movimento predito para ocultar os vetores de deslocamento em regiões estáticas ou que correspondem ao fundo (background).
* **Geração de Visualizações**: Gera gráficos contendo:
  * À esquerda: A nuvem de pontos LiDAR atual projetada em 2D.
  * À direita: O campo vetorial de predição de movimento futuro (desenho quiver), colorido conforme a categoria identificada (ex.: ônibus, pedestre, ciclista).
* **Geração do GIF**: Consolida as saídas salvas em uma animação de vídeo compactada (`result.gif`) facilitando a demonstração visual do comportamento dinâmico.
  * *Observação*: Para compilar vídeos mais longos em alta resolução a partir dos frames salvos `.png`, recomenda-se o uso do `ffmpeg`:
    ```bash
    ffmpeg -framerate 10 -i %d.png -c:v libx264 -pix_fmt yuv420p demonstracao_iara.mp4
    ```

### C. `data/test_carmen_parser.py`
* **Finalidade**: Um script de teste auxiliar criado para verificar de forma isolada a rotina de descompactação binária de arquivos individuais de nuvens de pontos `.pointcloud` do CARMEN, validando o arranjo de distâncias/intensidades antes de sua implementação na pipeline completa.

---

## 5. Resumo do Fluxo de Execução

O fluxo de processamento funciona em duas etapas principais, conforme documentado no arquivo `passo-a-passo.txt`:

1. **Conversão de Dados**:
   ```bash
   venv/bin/python MotionNet/data/carmen_to_motionnet.py \
     --log /caminho/do/seu_log_carmen.txt \
     --out_dir carmen-preprocessed-novo \
     --max_frames 100
   ```
2. **Inferência e Visualização**:
   ```bash
   PYTHONPATH=$(pwd)/MotionNet:$(pwd)/MotionNet/nuscenes-devkit/python-sdk \
   venv/bin/python MotionNet/test_carmen.py \
     --data_dir carmen-preprocessed-novo \
     --img_save_dir carmen_results_novo
   ```
   *(Essa etapa gera os plots individuais de cada frame e compila o vídeo final `result.gif`)*.

3. **Compilação de Vídeo MP4 (Opcional - via FFmpeg)**:
   Para compilar os frames individuais gerados em um vídeo MP4 de alta definição, navegue até a pasta onde as imagens `.png` foram salvas e execute:
   ```bash
   ffmpeg -framerate 10 -i %d.png -c:v libx264 -pix_fmt yuv420p demonstracao_iara.mp4
   ```
