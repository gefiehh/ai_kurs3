import re
import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
from pathlib import Path
import os

# Конфигурация
ASR_MODEL_SIZE = "base"  # tiny, base, small, medium, large
SAMPLE_RATE = 16000
TEMP_AUDIO_FILE = Path("temp_speech.wav")

# Параметры для записи с обнаружением тишины
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.5
MAX_RECORD_DURATION = 30
MIN_RECORD_DURATION = 0.5

# Загрузка модели Whisper
print(f"Загрузка модели Whisper ({ASR_MODEL_SIZE})...")
try:
    whisper_model = whisper.load_model(ASR_MODEL_SIZE)
    print(f"Модель Whisper {ASR_MODEL_SIZE} загружена")
except Exception as e:
    print(f"Ошибка загрузки Whisper: {e}")
    whisper_model = None


def is_silence(audio_chunk, threshold=SILENCE_THRESHOLD):
    return np.max(np.abs(audio_chunk)) < threshold


def record_audio_adaptive(filename: Path = TEMP_AUDIO_FILE):
    print("Говорите...")
    
    recorded_chunks = []
    silent_chunks = 0
    silent_chunks_needed = int(SILENCE_DURATION / 0.1)
    
    chunk_duration = 0.1
    chunk_samples = int(SAMPLE_RATE * chunk_duration)
    
    print("  (слушаю...)")
    
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
            recording = False
            total_duration = 0
            
            while True:
                data, overflowed = stream.read(chunk_samples)
                if overflowed:
                    print("Переполнение буфера")
                
                chunk = data.flatten()
                total_duration += chunk_duration
                
                if is_silence(chunk):
                    if recording:
                        silent_chunks += 1
                        if silent_chunks >= silent_chunks_needed:
                            print(f"  (тишина {SILENCE_DURATION} сек, останавливаюсь)")
                            break
                else:
                    if not recording:
                        print("  (слышу голос, записываю...)")
                        recording = True
                    silent_chunks = 0
                
                if recording:
                    recorded_chunks.append(chunk)
                
                if total_duration >= MAX_RECORD_DURATION:
                    print(f"  (достигнута максимальная длительность {MAX_RECORD_DURATION} сек)")
                    break
    except Exception as e:
        print(f"Ошибка при записи: {e}")
        return None
    
    if not recorded_chunks:
        print("Ничего не записано")
        return None
    
    # Объединяем все chunks
    audio_data = np.concatenate(recorded_chunks)
    
    # Проверяем минимальную длительность
    if len(audio_data) < MIN_RECORD_DURATION * SAMPLE_RATE:
        print(f"Запись слишком короткая ({len(audio_data)/SAMPLE_RATE:.1f} сек)")
        return None
    
    # Сохраняем как WAV файл
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    # Нормализуем и конвертируем в int16
    audio_int16 = (audio_data * 32767).astype('int16')
    write(str(filename), SAMPLE_RATE, audio_int16)
    
    if filename.exists():
        duration = len(audio_data) / SAMPLE_RATE
        file_size = filename.stat().st_size
        return str(filename)
    else:
        print(f"Не удалось сохранить файл {filename}")
        return None


def speech_to_text(filename: str = None) -> str:
    if filename is None:
        filename = str(TEMP_AUDIO_FILE)
    
    if whisper_model is None:
        print("Модель Whisper не загружена")
        return ""
    
    if not os.path.exists(filename):
        print(f"Файл не найден: {filename}")
        return ""
    
    try:
        # Загружаем аудио напрямую через scipy
        sample_rate, audio_data = read(filename)
        
        # Конвертируем в float32 и нормализуем
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32767.0
        elif audio_data.dtype == np.int32:
            audio_float = audio_data.astype(np.float32) / 2147483647.0
        else:
            audio_float = audio_data.astype(np.float32)
        
        # Если стерео, берем один канал
        if len(audio_float.shape) > 1:
            audio_float = audio_float.mean(axis=1)
        
        # Ресемплируем если нужно
        if sample_rate != SAMPLE_RATE:
            print(f"  (ресемплинг с {sample_rate} Гц на {SAMPLE_RATE} Гц)")
            # Простой ресемплинг через scipy
            from scipy import signal
            audio_float = signal.resample(audio_float, int(len(audio_float) * SAMPLE_RATE / sample_rate))
        
        # Распознаем
        result = whisper_model.transcribe(
            audio_float,
            language="ru",
            task="transcribe",
            fp16=False,
            verbose=False
        )
        return result["text"].strip()
        
    except Exception as e:
        print(f"Ошибка распознавания: {e}")
        import traceback
        traceback.print_exc()
        return ""


def clean_asr_text(text: str) -> str:
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r"[^а-яa-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def listen() -> str:
    if whisper_model is None:
        print("Голосовой ввод недоступен: модель Whisper не загружена")
        return ""
    
    try:
        # Адаптивная запись
        audio_file = record_audio_adaptive()
        
        if audio_file is None:
            print("Не удалось записать голос (тишина или слишком коротко)")
            return ""
        
        # Распознавание речи
        raw_text = speech_to_text(audio_file)
        
        if not raw_text:
            print("Не удалось распознать речь")
            return ""
        
        # Очистка текста
        cleaned_text = clean_asr_text(raw_text)
        
        # Удаляем временный файл
        try:
            Path(audio_file).unlink()
        except:
            pass
        
        return cleaned_text
        
    except Exception as e:
        print(f"Ошибка в listen(): {e}")
        import traceback
        traceback.print_exc()
        return ""