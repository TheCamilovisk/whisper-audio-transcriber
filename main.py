import logging
import tempfile
from datetime import datetime
from pathlib import Path

import torch
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Pipeline,
    pipeline,
)

from settings import get_settings

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)


async def download_audio(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> Path:
    audio = update.message.audio or update.message.voice
    if not audio:
        logging.warning('Nenhum arquivo de áudio encontrado na mensagem.')
        raise ValueError('Nenhum arquivo de áudio encontrado na mensagem.')

    file_id = audio.file_id

    file_ext = '.ogg' if update.message.voice else '.mp3'
    file_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + file_ext
    file_path = Path(tempfile.gettempdir()) / file_name

    logging.info('Baixando arquivo de áudio...')
    audio_file = await context.bot.get_file(file_id)
    await audio_file.download_to_drive(custom_path=str(file_path))
    logging.info(f'Arquivo de áudio salvo em: {file_path}')

    if not file_path.exists():
        logging.error(f'Erro: Arquivo {file_path} não encontrado.')
        raise RuntimeError('Erro ao baixar o arquivo de áudio.')

    return file_path


def transcribe_audio(file_path: Path, pipe: Pipeline, output_dir: Path) -> str:
    logging.info(f'Transcrevendo {file_path}...')
    result = pipe(str(file_path))
    transcription_text = result['text']

    output_file = output_dir / f'{file_path.stem}_transcription.txt'
    output_file.write_text(transcription_text, encoding='utf-8')

    logging.info(f'Transcrição salva em: {output_file}')

    return transcription_text


class AudioTranscriber:
    def __init__(self, model_name: str, output_dir: str, device: str):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.device = device

        self.processor = None
        self.model = None
        self.pipeline = self._init_pipeline()

    def _init_pipeline(self):
        torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32

        logging.info(f'Carregando modelo {self.model_name}...')
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation='flash_attention_2',
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        return pipeline(
            'automatic-speech-recognition',
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            torch_dtype=torch_dtype,
            device=self.device,
        )

    async def handle_audio(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        file_path = await download_audio(update, context)

        transcription_text = transcribe_audio(
            file_path=file_path,
            pipe=self.pipeline,
            output_dir=self.output_dir,
        )

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=transcription_text,
        )

        file_path.unlink()
        logging.info(f'Arquivo temporário removido: {file_path}')

    async def __call__(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        await self.handle_audio(update, context)


if __name__ == '__main__':
    settings = get_settings()
    application = ApplicationBuilder().token(settings.bot_token).build()

    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcriber = AudioTranscriber(
        model_name='openai/whisper-large-v3-turbo',
        output_dir=output_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    audio_handler = MessageHandler(filters.AUDIO | filters.VOICE, transcriber)
    application.add_handler(audio_handler)

    application.run_polling()
