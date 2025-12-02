# Whisper Audio Transcriber

Um script em Python para transcrição de áudio utilizando o modelo Whisper da OpenAI. Permite gravar áudio diretamente do microfone ou transcrever arquivos de áudio existentes.

## Objetivo

Este projeto oferece uma interface de linha de comando simples para:

- Gravar áudio diretamente do microfone
- Transcrever arquivos de áudio em texto
- Utilizar o modelo Whisper Large V3 Turbo para transcrições de alta qualidade
- Salvar automaticamente as transcrições em arquivos de texto

## Requisitos

- Python >= 3.13
- CUDA (opcional, para aceleração GPU)
- PortAudio (para gravação de áudio via PyAudio)

### Instalação do PortAudio

O PyAudio requer a biblioteca PortAudio instalada no sistema:

**Ubuntu/Debian:**

```bash
sudo apt-get install portaudio19-dev
```

**Fedora:**

```bash
sudo dnf install portaudio-devel
```

**macOS:**

```bash
brew install portaudio
```

**Arch Linux:**

```bash
sudo pacman -S portaudio
```

## Instalação

Este projeto utiliza [uv](https://github.com/astral-sh/uv) para gerenciamento de dependências.

1. Clone o repositório:

```bash
git clone https://github.com/TheCamilovisk/whisper-audio-transcriber.git
cd whisper-audio-transcriber
```

2. Instale as dependências:

```bash
uv sync
```

## Uso

### Gravar e transcrever áudio do microfone

```bash
uv run python main.py
```

O script irá:

1. Iniciar a gravação de áudio
2. Aguardar até você pressionar ENTER para parar
3. Transcrever o áudio gravado
4. Salvar a transcrição em `outputs/`

### Transcrever um arquivo de áudio existente

```bash
uv run python main.py arquivo.wav
```

### Especificar diretório de saída

```bash
uv run python main.py arquivo.wav /caminho/para/saida
```

## Dependências Principais

- **transformers** (4.47): Framework para modelos de IA, incluindo Whisper
- **torch** (>=2.9.1): PyTorch para computação de tensores
- **pyaudio** (>=0.2.14): Gravação de áudio do microfone
- **typer** (>=0.20.0): Interface de linha de comando
- **accelerate** (>=1.12.0): Otimização de modelos de IA
- **flash-attn**: Implementação otimizada de attention para GPUs
- **tokenizers** (0.21): Tokenização rápida

## Pacotes Adicionais

### Flash Attention

Este projeto utiliza uma versão pré-compilada do Flash Attention para melhor performance em GPUs. A wheel é baixada automaticamente de:

```
https://github.com/mjun0812/flash-attention-prebuild-wheels
```

Versão: flash_attn 2.8.1 (CUDA 12.8, torch 2.9, Python 3.13)

### Ferramentas de Desenvolvimento

- **ruff** (>=0.14.7): Linter e formatador de código Python

## Saída

As transcrições são salvas por padrão no diretório `outputs/` com o formato:

```
<nome_do_arquivo>_transcription.txt
```

Para gravações do microfone, o arquivo temporário é nomeado com timestamp:

```
YYYY-MM-DD_HH-MM-SS.wav
```

## Modelo

Utiliza o modelo **openai/whisper-large-v3-turbo** da Hugging Face, que oferece:

- Alta qualidade de transcrição
- Suporte multilíngue
- Otimização para velocidade (versão turbo)
- Compatibilidade com Flash Attention 2
