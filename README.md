# Whisper Audio Transcriber

Bot do Telegram para transcrição de áudio utilizando o modelo Whisper da OpenAI. Transcreve mensagens de áudio e voz automaticamente.

## Objetivo

Este projeto oferece um bot do Telegram que:

- Transcreve mensagens de áudio e voz automaticamente
- Utiliza o modelo Whisper Large V3 Turbo para transcrições de alta qualidade
- Salva automaticamente as transcrições em arquivos de texto
- Responde com o texto transcrito diretamente no chat

## Requisitos

- Python >= 3.13
- CUDA (opcional, para aceleração GPU)
- Token de bot do Telegram (obtido via [@BotFather](https://t.me/botfather))

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

3. Configure as variáveis de ambiente:

Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:

```env
BOT_TOKEN=seu_token_aqui
OUTPUT_DIR=outputs  # opcional, padrão: outputs/
```

Para obter um token do bot:

1. Converse com [@BotFather](https://t.me/botfather) no Telegram
2. Use o comando `/newbot` e siga as instruções
3. Copie o token fornecido e cole no arquivo `.env`

## Uso

Execute o bot:

```bash
uv run python main.py
```

O bot ficará online aguardando mensagens. Para usar:

1. Inicie uma conversa com seu bot no Telegram
2. Envie uma mensagem de áudio ou voz
3. Aguarde a transcrição (o bot responderá com o texto)

As transcrições também são salvas automaticamente no diretório configurado (padrão: `outputs/`).

## Dependências Principais

- **transformers** (4.47): Framework para modelos de IA, incluindo Whisper
- **torch** (>=2.9.1): PyTorch para computação de tensores
- **python-telegram-bot** (>=22.5): Framework para bots do Telegram
- **pydantic-settings** (>=2.12.0): Gerenciamento de configurações
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

As transcrições são salvas automaticamente no diretório configurado (padrão: `outputs/`) com o formato:

```
<nome_do_arquivo>_transcription.txt
```

Os arquivos de áudio baixados do Telegram são nomeados com timestamp:

```
YYYY-MM-DD_HH-MM-SS.ogg  # para mensagens de voz
YYYY-MM-DD_HH-MM-SS.mp3  # para mensagens de áudio
```

Os arquivos temporários são removidos automaticamente após a transcrição.

## Modelo

Utiliza o modelo **openai/whisper-large-v3-turbo** da Hugging Face, que oferece:

- Alta qualidade de transcrição
- Suporte multilíngue
- Otimização para velocidade (versão turbo)
- Compatibilidade com Flash Attention 2
