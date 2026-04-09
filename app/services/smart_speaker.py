import os
import torch
import numpy as np
import sounddevice as sd
import queue
import time
import gc
import sys
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from faster_whisper import WhisperModel
from PIL import Image

class SmartSpeakerOnDevice:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio_queue = queue.Queue(maxsize=100)
        self.is_listening = True
        self.consecutive_errors = 0
        self.max_errors = 5
        
        print(f"🚀 Initializing SmartSpeaker on {self.device} (8GB VRAM Target)")
        
        # 1. STT 모델 로드 (상시 가동, INT8로 가볍게)
        print("🎙️ Loading STT (openai/whisper-large-v3 via faster-whisper)...")
        try:
            self.stt_model = WhisperModel(
                "large-v3", 
                device=self.device, 
                compute_type="int8" if self.device == "cuda" else "int8"
            )
        except Exception as e:
            print(f"Failed to load STT model: {e}")
            sys.exit(1)
        
        # 2. TTS 모델 로드 (상시 가동)
        print("🔊 Loading TTS (Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)...")
        self.tts_model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        try:
            self.tts_tokenizer = AutoTokenizer.from_pretrained(self.tts_model_name, trust_remote_code=True)
            self.tts_model = AutoModelForCausalLM.from_pretrained(
                self.tts_model_name, 
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).to(self.device)
            self.tts_model.eval()
        except Exception as e:
            print(f"⚠️ TTS Load Error (비프음으로 대체됩니다): {e}")
            self.tts_model = None
        
        # 3. OCR 모델 (필요할 때만 로드하기 위해 변수만 선언)
        self.ocr_model_name = "zai-org/GLM-OCR"
        self.ocr_model = None
        self.ocr_tokenizer = None
        
        print("✅ 초기화 완료! 실시간 대기 모드에 진입합니다.")

    # -----------------------------------------------------------------
    # 메모리 관리 & VLM (비전 모델) 로직
    # -----------------------------------------------------------------
    def load_ocr_model(self):
        """메모리 절약을 위해 필요할 때만 GLM-OCR을 4-bit로 로드"""
        if self.ocr_model is None:
            print("👁️ Lazy Loading GLM-OCR model (4-bit quantization)...")
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.ocr_tokenizer = AutoTokenizer.from_pretrained(self.ocr_model_name, trust_remote_code=True)
                self.ocr_model = AutoModelForCausalLM.from_pretrained(
                    self.ocr_model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.ocr_model.eval()
            except Exception as e:
                print(f"Error loading OCR model: {e}")
                self.unload_ocr_model()
                raise
            
    def unload_ocr_model(self):
        """약물 인식 완료 후 VRAM 즉시 반환"""
        if self.ocr_model is not None:
            print("🧹 Unloading GLM-OCR to free VRAM...")
            del self.ocr_model
            del self.ocr_tokenizer
            self.ocr_model = None
            self.ocr_tokenizer = None
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def process_ocr(self, image_path: str):
        """이미지에서 약물 정보 추출 (GLM-OCR)"""
        try:
            self.load_ocr_model()
            print(f"📸 이미지를 분석 중입니다... ({image_path})")
            
            # 실제 카메라 연동 시 이 부분을 수정하세요.
            if not os.path.exists(image_path):
                return "테스트용 이미지를 찾을 수 없습니다."

            image = Image.open(image_path).convert('RGB')
            prompt = "이 이미지에 있는 약의 이름과 성분을 정확하게 알려주세요."
            
            inputs = self.ocr_tokenizer(prompt, image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.ocr_model.generate(**inputs, max_new_tokens=100)
                
            result_text = self.ocr_tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"🔍 OCR 분석 결과: {result_text}")
            
        except Exception as e:
            print(f"OCR Processing Error: {e}")
            result_text = "죄송합니다. 약물 인식 중 오류가 발생했습니다."
        finally:
            self.unload_ocr_model() # 에러가 나도 무조건 메모리 반환
            
        return result_text

    # -----------------------------------------------------------------
    # 음성 입출력 & 실시간 처리 로직
    # -----------------------------------------------------------------
    def synthesize_speech(self, text: str):
        """Qwen3-TTS를 이용한 음성 합성"""
        print(f"🗣️ AI 스피커 발화: {text}")
        if self.tts_model is None:
            # 모델 로드 실패 시 가짜 비프음 생성 (디버그용)
            sample_rate = 24000
            t = np.linspace(0, 1.0, int(sample_rate * 1.0), False)
            return np.sin(t * 440) * 0.5, sample_rate

        try:
            with torch.no_grad():
                inputs = self.tts_tokenizer(text, return_tensors="pt").to(self.device)
                audio_output = self.tts_model.generate(**inputs)
                audio_data = audio_output.cpu().numpy().squeeze()
                sample_rate = 24000
            return audio_data, sample_rate
        except Exception as e:
            print(f"TTS Error: {e}")
            return None, None

    def play_audio(self, audio_data, sample_rate):
        """스피커로 재생"""
        if audio_data is None or sample_rate is None:
            return
        try:
            sd.play(audio_data, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Audio Playback Error: {e}")

    def audio_callback(self, indata, frames, time_info, status):
        """마이크 입력 콜백 - 1D Numpy 배열로 큐에 삽입"""
        if status:
            pass
        try:
            if self.audio_queue.full():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
            self.audio_queue.put_nowait(indata[:, 0].copy())
        except Exception as e:
            print(f"Audio Callback Error: {e}")

    def process_stt(self, audio_data):
        """모인 음성 데이터로 Whisper 추론"""
        try:
            segments, info = self.stt_model.transcribe(audio_data, beam_size=5, language="ko")
            transcribed_text = "".join([segment.text for segment in segments]).strip()
            
            if len(transcribed_text) > 1: # 노이즈 무시
                print(f"👤 사용자: {transcribed_text}")
                self.handle_user_input(transcribed_text)
            
            self.consecutive_errors = 0
        except Exception as stt_err:
            print(f"STT Processing Error: {stt_err}")
            self.consecutive_errors += 1

    def start_listening(self):
        """실시간 음성 감지(VAD) 기반 메인 루프"""
        sample_rate = 16000
        print("\n👂 마이크 듣는 중... (종료: Ctrl+C)")
        
        # --- VAD (음성 감지) 튜닝 설정 ---
        energy_threshold = 0.005      # 마이크 환경에 따라 0.001 ~ 0.01 사이로 조절하세요
        max_silence_duration = 0.8    # 0.8초 이상 조용하면 문장 끝으로 간주
        min_speech_duration = 0.5     # 0.5초 이하의 짧은 소리는 무시 (헛기침 등)
        # -------------------------------

        while self.is_listening:
            try:
                with sd.InputStream(samplerate=sample_rate, channels=1, callback=self.audio_callback):
                    audio_buffer = []
                    is_recording = False
                    silence_timer = 0.0
                    
                    while self.is_listening:
                        try:
                            # 큐에서 오디오 조각 꺼내기
                            data = self.audio_queue.get(timeout=0.1)
                            energy = np.mean(data**2)

                            if energy > energy_threshold:
                                if not is_recording:
                                    print("\n[🎙️ 발화 감지됨! 듣는 중...]")
                                    is_recording = True
                                
                                audio_buffer.append(data)
                                silence_timer = 0.0 
                                
                            elif is_recording:
                                audio_buffer.append(data)
                                chunk_duration = len(data) / sample_rate
                                silence_timer += chunk_duration
                                
                                # 침묵이 길어지면 녹음 종료 및 STT 처리
                                if silence_timer > max_silence_duration:
                                    is_recording = False
                                    full_audio = np.concatenate(audio_buffer, axis=0)
                                    audio_buffer = []
                                    
                                    total_duration = len(full_audio) / sample_rate
                                    if total_duration > min_speech_duration:
                                        print("[⏳ 발화 종료. 의미 분석 중...]")
                                        self.process_stt(full_audio)
                                    else:
                                        print("[노이즈 무시됨]")
                                        
                        except queue.Empty:
                            pass 
                            
            except KeyboardInterrupt:
                print("사용자 요청으로 종료합니다.")
                self.is_listening = False
                break
            except Exception as e:
                print(f"Loop Error: {e}. 2초 후 재시도...")
                time.sleep(2)
                self.consecutive_errors += 1
                
            if self.consecutive_errors >= self.max_errors:
                print("CRITICAL: 에러 누적. 시스템 컴포넌트를 재시작합니다...")
                self.restart_system()
                self.consecutive_errors = 0

    def restart_system(self):
        """심각한 에러 시 메모리 비우고 STT 재초기화"""
        print("긴급 메모리 초기화 실행 중...")
        self.unload_ocr_model()
        
        try:
            print("STT 모델 재시작...")
            del self.stt_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.stt_model = WhisperModel("large-v3", device=self.device, compute_type="int8")
        except Exception as e:
            print(f"STT 복구 실패: {e}")
            
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        time.sleep(2)

    def handle_user_input(self, text: str):
        """인식된 텍스트를 바탕으로 분기 처리"""
        # "약", "이거 뭐야" 등의 키워드가 포함되어 있으면 OCR 로직 실행
        if any(keyword in text for keyword in ["약", "이거 뭐야", "무슨 약", "찍어줘"]):
            print("[Action] 카메라 찰칵! (약물 인식 트리거 발동)")
            
            # TODO: 실제로는 여기에 cv2 웹캠 촬영 코드가 들어가야 합니다.
            # 지금은 더미 이미지가 있다고 가정합니다.
            image_path = "dummy_drug_image.jpg" 
            
            # 테스트를 위해 임의의 이미지를 하나 만들어 두시는 게 좋습니다.
            if not os.path.exists(image_path):
                # 테스트 환경을 위해 파일이 없으면 그냥 빈 이미지 파일 하나 생성 (에러 방지용)
                Image.new('RGB', (100, 100), color = 'white').save(image_path)
                
            drug_info = self.process_ocr(image_path)
            response_text = f"이 약은 {drug_info} 로 보입니다."
        else:
            response_text = "네, 듣고 있어요."

        # 응답 음성 합성 및 재생
        audio_data, sr = self.synthesize_speech(response_text)
        self.play_audio(audio_data, sr)

if __name__ == "__main__":
    # OS 레벨에서 메모리 단편화 방지 (VRAM 8GB 활용 극대화)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    while True:
        try:
            print("\n" + "="*50)
            print("AI 약물 인식 스피커 서비스 시작...")
            print("="*50)
            speaker = SmartSpeakerOnDevice()
            speaker.start_listening()
            
            if not speaker.is_listening:
                break
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"FATAL SYSTEM ERROR: {e}")
            traceback.print_exc()
            print("5초 후 전체 시스템을 재시작합니다...")
            time.sleep(5)