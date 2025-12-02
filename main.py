import tempfile
import threading
import wave
from pathlib import Path

import pyaudio
import torch
import typer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

app = typer.Typer()


def record_audio() -> str:
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    temp_dir = Path(tempfile.gettempdir())
    OUTPUT_FILENAME = str(temp_dir / 'recorded_audio.wav')

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    typer.echo('Gravando... (Pressione ENTER para parar)')
    frames = []
    recording = threading.Event()
    recording.set()

    def listen_for_stop():
        input()
        recording.clear()

    listener_thread = threading.Thread(target=listen_for_stop, daemon=True)
    listener_thread.start()

    try:
        while recording.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
    except KeyboardInterrupt:
        recording.clear()

    typer.echo('Gravação finalizada.')

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return OUTPUT_FILENAME


@app.command()
def transcribe(
    audio_file: Path = typer.Argument(
        None, help='Arquivo de áudio a ser transcrito'
    ),
    output_dir: Path = typer.Argument(
        None, help='Diretório onde será salvo o resultado da transcrição'
    ),
):
    """
    Transcreve um arquivo de áudio usando o modelo Whisper.
    """
    temp_audio_file = None
    if audio_file is None:
        typer.echo('Gravando àudio do microfone..')
        temp_audio_file = record_audio()
        audio_file = Path(temp_audio_file)

    if not audio_file.exists():
        typer.echo(f'Erro: Arquivo {audio_file} não encontrado.')
        raise typer.Exit(code=1)

    if output_dir is None:
        root_folder = Path(__file__).parent
        output_dir = root_folder / 'outputs'

    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.float16 if device == 'cuda' else torch.float32

    model_name = 'openai/whisper-large-v3-turbo'

    typer.echo(f'Carregando modelo {model_name}...')
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation='flash_attention_2',
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_name)

    pipe = pipeline(
        'automatic-speech-recognition',
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )

    typer.echo(f'Transcrevendo {audio_file}...')
    result = pipe(str(audio_file))
    transcription_text = result['text']

    output_file = output_dir / f'{audio_file.stem}_transcription.txt'
    output_file.write_text(transcription_text, encoding='utf-8')

    typer.echo(f'Transcrição salva em: {output_file}')
    typer.echo(f'\nTexto transcrito:\n{transcription_text}')

    if temp_audio_file is not None:
        Path(temp_audio_file).unlink()
        typer.echo(f'Arquivo temporário removido: {temp_audio_file}')


if __name__ == '__main__':
    app()
