# voice_service.py
import hashlib
import re
import threading
from pathlib import Path
from typing import Optional
import torch
import scipy
import pygame

from transformers import VitsModel, AutoTokenizer
from ruaccent import RUAccent


class VoiceService:
    #Сервис для озвучивания через VITS (UtrobinTTS)
    
    def __init__(self, language: str = "ru"):
        self.language = language
        self.cache_dir = Path("tts_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._enabled = False
        self.speaker_id = 0  
        
        # Загрузка модели VITS
        try:
            model_name = "utrobinmv/tts_ru_free_hf_vits_low_multispeaker"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = VitsModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model.eval()
            
            # Загрузка ударений
            self.accentizer = RUAccent()
            self.accentizer.load(omograph_model_size='turbo', use_dictionary=True)
            
            self._enabled = True
            
            # Инициализация pygame
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            print(f"VITS TTS инициализирован (голос: {'мужской' if self.speaker_id == 1 else 'женский'})")
        except Exception as e:
            print(f"Ошибка инициализации VITS: {e}")
    
    def _normalize_text(self, text: str) -> str:
        # Расстановка ударений
        text = self.accentizer.process_all(text)
        return text
    
    def _get_cache_path(self, text: str) -> Path:
        #Генерация пути для кэшированного аудио
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return self.cache_dir / f"{text_hash}.wav"
    
    def _speak_sync(self, text: str):
        #Синхронное озвучивание
        if not self._enabled:
            return
        
        try:
            normalized = self._normalize_text(text)
            cache_path = self._get_cache_path(normalized)
            
            if not cache_path.exists():
                # Генерация аудио
                inputs = self.tokenizer(normalized, return_tensors="pt").to(self.device)

                
                with torch.no_grad():
                    output = self.model(**inputs, speaker_id=self.speaker_id).waveform
                    output = output.detach().cpu().numpy()
                
                # Сохранение
                scipy.io.wavfile.write(
                    str(cache_path), 
                    rate=self.model.config.sampling_rate,
                    data=output[0]
                )
            
            # Воспроизведение
            pygame.mixer.music.load(str(cache_path))
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                threading.Event().wait(0.1)
                
        except Exception as e:
            print(f"Ошибка VITS: {e}")
    
    def speak(self, text: str, async_mode: bool = True):
        #Озвучивание текста
        if not text or not self._enabled:
            return
        
        if async_mode:
            thread = threading.Thread(
                target=self._speak_sync,
                args=(text,),
                daemon=True
            )
            thread.start()
        else:
            self._speak_sync(text)

    def set_voice(self, voice: str):
        #Смена голоса: 'woman' или 'man'
        self.speaker_id = 0 if voice == "woman" else 1
    
    def is_enabled(self) -> bool:
        return self._enabled