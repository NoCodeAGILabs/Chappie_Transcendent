"""
CHAPPIE NEURAL HYBRID v2.0 - Sistema Publicable
===============================================
Arquitectura Tri-Cameral con Meta-Learning y Nested Learning
Autor: Víctor
Componentes Científicos:
- Cerebro Biológico (Nengo SNN)
- Nested Learning (LLM → SNN)
- Meta-Learning (Auto-optimización)
- Visión Multi-Modal (Caras + Objetos + Dígitos)
- Speech Recognition Avanzado (Whisper)
"""

import threading
import queue
import time
import pickle
import os
import signal
import sys
import json
import logging
import warnings
import subprocess as sp
from datetime import datetime
from collections import deque
from pathlib import Path

import numpy as np
import cv2
import pyttsx3
import speech_recognition as sr
import whisper
import ollama
import nengo
import nengo_dl
import tensorflow as tf
import face_recognition
import imageio_ffmpeg

# =============================================================================
# CONFIGURACIÓN FFMPEG PARA WHISPER
# =============================================================================
warnings.filterwarnings("ignore")

# Detectar Sistema Operativo
IS_WINDOWS = sys.platform == "win32"

# Configuración dinámica de FFmpeg
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
FFMPEG_DIR = os.path.dirname(FFMPEG_EXE)

# Extensiones
EXE_EXT = ".exe" if IS_WINDOWS else ""
FFPROBE_EXE = os.path.join(FFMPEG_DIR, f"ffprobe{EXE_EXT}")

# Fallbacks
if not os.path.exists(FFPROBE_EXE): 
    FFPROBE_EXE = "ffprobe"

# Agregar al PATH
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ["PATH"]
os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_EXE
os.environ["IMAGEIO_FFPROBE_EXE"] = FFPROBE_EXE

# Parche para subprocess
_Original_Popen = sp.Popen

def patched_popen(cmd, *a, **kw):
    if isinstance(cmd, (list, tuple)) and len(cmd) > 0:
        cmd = list(cmd)
        prog = str(cmd[0]).lower()
        if "ffmpeg" in prog:
            cmd[0] = FFMPEG_EXE
    elif isinstance(cmd, str):
        if "ffmpeg" in cmd.lower():
            cmd = cmd.replace("ffmpeg", FFMPEG_EXE, 1)
    
    if "executable" in kw:
        if "ffmpeg" in str(kw["executable"]).lower():
            kw["executable"] = FFMPEG_EXE
    
    return _Original_Popen(cmd, *a, **kw)

sp.Popen = patched_popen

print(f"✓ FFmpeg configurado: {FFMPEG_EXE}")

# =============================================================================
# CONFIGURACIÓN CIENTÍFICA
# =============================================================================
MODELO_LLM = "tinyllama:latest"
MODELO_WHISPER = "base"  # "small" para mejor español, pero más lento
ARCHIVO_MEMORIA = "chappie_memoria_v2.pkl"
ARCHIVO_CARAS = "chappie_faces.pkl"
NOMBRE_CREADOR = "Víctor"

# Directorio de experimentos
EXP_DIR = Path("experimentos")
EXP_DIR.mkdir(exist_ok=True)

# =============================================================================
# LOGGING CIENTÍFICO
# =============================================================================
class MetricsLogger:
    """Logger para análisis científico y publicación"""
    def __init__(self, experimento_id=None):
        self.exp_id = experimento_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = EXP_DIR / f"metricas_{self.exp_id}.jsonl"
        self.metricas = []
        
        # Configurar logging con UTF-8 para Windows
        log_filename = EXP_DIR / f"sistema_{self.exp_id}.log"
        
        # Handler de archivo con UTF-8
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Handler de consola con UTF-8
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formato sin emojis problemáticos
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Configurar root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler],
            force=True  # Sobrescribe configuración previa
        )
        
        self.logger = logging.getLogger("CHAPPIE")
    
    def log(self, timestamp, cerebro_state, lr_state, evento=None, extra=None):
        entrada = {
            "t": timestamp,
            "motor": float(cerebro_state.get('motor', 0)),
            "emocion": float(cerebro_state.get('emocion', 0)),
            "cognitivo": float(cerebro_state.get('cognitivo', 0)),
            "error": float(cerebro_state.get('error', 0)),
            "lr_motor": float(lr_state.get('motor', 0)),
            "lr_emocion": float(lr_state.get('emocion', 0)),
            "lr_cognitivo": float(lr_state.get('cognitivo', 0)),
            "evento": evento,
            "extra": extra
        }
        self.metricas.append(entrada)
        
        # Guardado incremental
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entrada) + '\n')
    
    def generar_reporte(self):
        """Genera DataFrame para análisis"""
        try:
            import pandas as pd
            return pd.DataFrame(self.metricas)
        except ImportError:
            return self.metricas

# =============================================================================
# META-LEARNING CONFIG
# =============================================================================
class MetaLearningConfig:
    """Configuraciones experimentales para tesis"""
    def __init__(self, modo="conservador"):
        self.modos = {
            "conservador": {
                'lr_min': 1e-5, 'lr_max': 5e-4,
                'factor_aumento': 1.05, 'factor_disminucion': 0.95,
                'umbral_error_alto': 0.5, 'umbral_error_bajo': 0.2,
                'intervalo_ajuste': 100  # ciclos
            },
            "agresivo": {
                'lr_min': 1e-4, 'lr_max': 1e-2,
                'factor_aumento': 1.15, 'factor_disminucion': 0.85,
                'umbral_error_alto': 0.6, 'umbral_error_bajo': 0.15,
                'intervalo_ajuste': 50
            },
            "adaptativo": {  # LLM controla TODO
                'lr_min': 1e-6, 'lr_max': 1e-2,
                'factor_aumento': None,
                'factor_disminucion': None,
                'umbral_error_alto': 0.5, 'umbral_error_bajo': 0.2,
                'intervalo_ajuste': 80
            }
        }
        self.config = self.modos.get(modo, self.modos["conservador"])
        self.modo = modo

# =============================================================================
# 1. PERCEPCIÓN AVANZADA (INPUTS)
# =============================================================================
class VisionMultiModal:
    """Sistema de visión con reconocimiento facial, objetos y dígitos"""
    def __init__(self, logger):
        self.logger = logger.logger
        self.lock = threading.Lock()
        
        # Estados sensoriales
        self.luz = 0.5
        self.movimiento = 0.0
        self.persona = None
        self.objeto_detectado = None
        self.digito_detectado = None
        
        # Persistencia
        self.caras = {}
        self.aprender_nombre = None
        self.frame_previo = None
        self.activo = True
        
        # Red neuromorfica para dígitos
        self.digitos_net = None
        self.digitos_sim = None
        self._cargar_datos()
        self._entrenar_digitos()
        
        # Thread de visión
        threading.Thread(target=self._run, daemon=True).start()
    
    def _cargar_datos(self):
        """Carga caras conocidas"""
        if os.path.exists(ARCHIVO_CARAS):
            try:
                self.caras = pickle.load(open(ARCHIVO_CARAS, "rb"))
                self.logger.info(f"✓ Caras cargadas: {list(self.caras.keys())}")
            except Exception as e:
                self.logger.error(f"Error cargando caras: {e}")
    
    def _entrenar_digitos(self):
        """Entrena red neuromorfica para reconocer dígitos escritos a mano"""
        self.logger.info("🧠 Entrenando red neuromorfica (MNIST)...")
        try:
            (train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
            train_images = train_images.reshape((train_images.shape[0], -1))[:5000]
            train_images = train_images[:, None, :]
            train_labels = tf.keras.utils.to_categorical(train_labels)[:5000]
            train_labels = train_labels[:, None, :]
            
            # Modelo simple
            inp = tf.keras.Input(shape=(784,))
            x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(inp)
            out = tf.keras.layers.Dense(10)(x)
            model = tf.keras.Model(inputs=inp, outputs=out)
            
            # Conversión a SNN
            converter = nengo_dl.Converter(
                model, 
                swap_activations={tf.nn.relu: nengo.SpikingRectifiedLinear()}
            )
            
            with nengo_dl.Simulator(converter.net, minibatch_size=200) as sim:
                sim.compile(
                    optimizer=tf.optimizers.Adam(0.001),
                    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=[tf.metrics.CategoricalAccuracy()]
                )
                sim.fit(train_images, train_labels, epochs=1, verbose=0)
                
                self.digitos_net = converter.net
                self.digitos_sim = sim
                self.digitos_output = out
                self.digitos_converter = converter
            
            self.logger.info("✓ Red neuromorfica lista")
        except Exception as e:
            self.logger.error(f"Error entrenando dígitos: {e}")
    
    def get_estado(self):
        """Acceso thread-safe"""
        with self.lock:
            return {
                'luz': self.luz,
                'movimiento': self.movimiento,
                'persona': self.persona,
                'objeto': self.objeto_detectado,
                'digito': self.digito_detectado
            }
    
    def aprender(self, nombre):
        """Solicita aprendizaje de nueva cara"""
        self.aprender_nombre = nombre
    
    def reconocer_digito(self, frame):
        """Reconoce dígito usando red neuromorfica"""
        if self.digitos_sim is None:
            return None
        
        try:
            # Preprocesamiento
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
            resized = cv2.resize(thresh, (28, 28))
            
            # Predicción
            flat_data = resized.reshape(-1)
            n_steps = 30
            test_data = np.tile(flat_data, (1, n_steps, 1))
            batch_data = np.tile(test_data, (200, 1, 1))
            
            prediction = self.digitos_sim.predict(batch_data)
            output_spikes = prediction[self.digitos_converter.outputs[self.digitos_output]][0]
            sum_spikes = np.sum(output_spikes, axis=0)
            
            return int(np.argmax(sum_spikes))
        except Exception as e:
            self.logger.error(f"Error reconociendo dígito: {e}")
            return None
    
    def _run(self):
        """Loop principal de visión"""
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        
        self.logger.info("👁️ Sistema de visión activo")
        
        while self.activo:
            try:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # 1. Luz y Movimiento
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                luz_new = np.mean(gray) / 255.0
                
                mov_new = 0.0
                if self.frame_previo is not None:
                    delta = cv2.absdiff(self.frame_previo, gray)
                    mov_new = np.mean(delta) / 255.0
                
                self.frame_previo = gray.copy()
                
                # 2. Reconocimiento Facial
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                small_frame = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
                locs = face_recognition.face_locations(small_frame)
                
                # Aprendizaje de cara
                if self.aprender_nombre and locs:
                    enc = face_recognition.face_encodings(small_frame, locs)[0]
                    self.caras[self.aprender_nombre] = enc
                    with open(ARCHIVO_CARAS, "wb") as f:
                        pickle.dump(self.caras, f)
                    self.logger.info(f"✓ Cara guardada: {self.aprender_nombre}")
                    self.aprender_nombre = None
                
                # Identificación
                nombre_det = None
                if locs:
                    encs = face_recognition.face_encodings(small_frame, locs)
                    for enc in encs:
                        for n, d in self.caras.items():
                            if face_recognition.compare_faces([d], enc, tolerance=0.55)[0]:
                                nombre_det = n
                                break
                
                # 3. Detección de objetos simple (por color - ejemplo)
                objeto_det = self._detectar_objeto(frame)
                
                # Actualización thread-safe
                with self.lock:
                    self.luz = luz_new
                    self.movimiento = mov_new
                    self.persona = nombre_det
                    self.objeto_detectado = objeto_det
                
                time.sleep(0.15)
                
            except Exception as e:
                self.logger.error(f"Error en visión: {e}")
                time.sleep(0.1)
        
        cap.release()
    
    def _detectar_objeto(self, frame):
        """Detección simple de objetos por color (ejemplo: objeto rojo)"""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Rango para rojo
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            mask = cv2.inRange(hsv, lower_red, upper_red)
            
            if np.sum(mask) > 5000:  # Umbral de píxeles rojos
                return "objeto_rojo"
            
            return None
        except:
            return None

# =============================================================================
# 2. AUDIO AVANZADO
# =============================================================================
class OidoWhisper:
    """Sistema de audio con Whisper para español robusto"""
    def __init__(self, logger):
        self.logger = logger.logger
        self.q = queue.Queue()
        self.rec = sr.Recognizer()
        self.mic = sr.Microphone()
        
        # Cargar Whisper
        self.logger.info(f"[AUDIO] Cargando Whisper ({MODELO_WHISPER})...")
        self.whisper_model = whisper.load_model(MODELO_WHISPER)
        self.logger.info("[OK] Whisper listo")
        
        threading.Thread(target=self._run, daemon=True).start()
    
    def _run(self):
        with self.mic as s:
            self.rec.adjust_for_ambient_noise(s, duration=1)
        
        self.logger.info("[AUDIO] Oido Whisper activo")
        
        while True:
            try:
                with self.mic as s:
                    audio = self.rec.listen(s, timeout=None, phrase_time_limit=8)
                
                # Guardar temporalmente
                temp_file = "temp_audio.wav"
                with open(temp_file, "wb") as f:
                    f.write(audio.get_wav_data())
                
                # Transcripción Whisper
                result = self.whisper_model.transcribe(temp_file, language="es")
                texto = result["text"].strip()
                
                if texto:
                    self.logger.info(f"👤 Usuario: {texto}")
                    self.q.put(texto)
                
                # Limpiar
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
            except Exception as e:
                self.logger.error(f"Error en audio: {e}")
                time.sleep(0.5)

class Boca:
    """Sistema de síntesis de voz (Corregido para evitar bloqueos)"""
    def __init__(self, logger):
        self.logger = logger.logger
        self.q = queue.Queue()
        # Iniciamos el hilo, pero el motor se crea ADENTRO del hilo
        threading.Thread(target=self._run, daemon=True).start()
    
    def _run(self):
        # Inicialización DENTRO del hilo para evitar conflictos de Windows COM
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 155)
            
            # Buscar voz en español
            voices = engine.getProperty('voices')
            for v in voices:
                if "spanish" in v.name.lower() or "es-" in v.id.lower():
                    engine.setProperty('voice', v.id)
                    break
        except Exception as e:
            self.logger.error(f"Error fatal iniciando motor de voz: {e}")
            return

        while True:
            texto = self.q.get()
            if texto is None: break # Señal de apagado
            
            try:
                # En Windows, a veces hay que 'resetear' el loop
                engine.say(texto)
                engine.runAndWait()
            except Exception as e:
                self.logger.error(f"Error hablando: {e}")
                # Intento de recuperación simple
                try:
                    engine = pyttsx3.init()
                except:
                    pass
            
            self.q.task_done()
            time.sleep(0.1) # Pequeña pausa para dejar respirar al audio
    
    def decir(self, texto):
        self.logger.info(f"[CHAPPIE] {texto}")
        self.q.put(texto)

# =============================================================================
# 3. CEREBRO CON NESTED LEARNING
# =============================================================================
class CerebroNestedLearning:
    """
    Cerebro tri-cameral con:
    - Meta-Learning: LLM optimiza hiperparámetros
    - Nested Learning: LLM ajusta pesos internos de la SNN
    - PROTECCIÓN THREAD-SAFE (Corrección del error NoneType)
    """
    def __init__(self, vision, meta_config, logger):
        self.vision = vision
        self.meta_config = meta_config
        self.logger = logger.logger
        self.metrics = logger
        
        # Learning Rates
        self.lr = {
            'motor': 3e-4,
            'emocion': 2e-4,
            'cognitivo': 2e-4
        }
        
        self.estados = {'motor': 0, 'emocion': 0, 'cognitivo': 0, 'error': 0}
        self.lock = threading.Lock()     # Para variables de estado
        self.brain_lock = threading.Lock() # <--- NUEVO: Para el simulador Nengo
        
        # Colas de optimización
        self.meta_ajustes = queue.Queue()
        self.nested_ajustes = queue.Queue()
        
        # Historia para análisis
        self.historia_error = deque(maxlen=10)
        
        self._construir_red()
        
        # Threads de aprendizaje
        threading.Thread(target=self._meta_learning_loop, daemon=True).start()
        threading.Thread(target=self._nested_learning_loop, daemon=True).start()
    
    def _construir_red(self):
        """Red neuronal spiking tri-cameral"""
        # Reducimos neuronas a 100 para optimizar CPU
        self.net = nengo.Network(label="Cerebro Tri-Cameral v2")
        with self.net:
            # Nodos de entrada
            def func_mov(t):
                estado = self.vision.get_estado()
                return estado['movimiento'] * 5.0
            
            def func_emo(t):
                estado = self.vision.get_estado()
                base = (1.0 - estado['luz'])
                soc = -0.8 if estado['persona'] == NOMBRE_CREADOR else 0.2
                # Bajamos el bonus de objeto para reducir hiper-excitabilidad
                obj_bonus = 0.1 if estado['objeto'] else 0.0 
                return base + soc + obj_bonus
            
            def func_cog(t):
                return np.sin(t * 0.5) * 0.5
            
            node_mov = nengo.Node(func_mov)
            node_emo = nengo.Node(func_emo)
            node_cog = nengo.Node(func_cog)
            
            # Cortezas (Optimizadas para CPU: 100 neuronas)
            self.ens_motor = nengo.Ensemble(100, 1, radius=2.0, label="Motor")
            self.ens_emocion = nengo.Ensemble(100, 1, radius=2.0, label="Emocion")
            self.ens_cognitivo = nengo.Ensemble(100, 1, radius=2.0, label="Cognitivo")
            
            # Conexiones con aprendizaje PES
            nengo.Connection(node_mov, self.ens_motor)
            nengo.Connection(node_emo, self.ens_emocion)
            nengo.Connection(node_cog, self.ens_cognitivo)
            
            # Plasticidad (PES)
            self.conn_motor = nengo.Connection(
                self.ens_motor, self.ens_motor,
                learning_rule_type=nengo.PES(self.lr['motor'])
            )
            self.conn_emocion = nengo.Connection(
                self.ens_emocion, self.ens_emocion,
                learning_rule_type=nengo.PES(self.lr['emocion'])
            )
            
            # Señal de error para PES
            self.error_node = nengo.Node(lambda t: np.random.normal(0, 0.1))
            nengo.Connection(self.error_node, self.conn_motor.learning_rule)
            nengo.Connection(self.error_node, self.conn_emocion.learning_rule)
            
            # Probes
            self.p_motor = nengo.Probe(self.ens_motor, synapse=0.1)
            self.p_emocion = nengo.Probe(self.ens_emocion, synapse=0.1)
            self.p_cognitivo = nengo.Probe(self.ens_cognitivo, synapse=0.1)
        
        self.sim = nengo.Simulator(self.net, progress_bar=False)
        self.logger.info("[CEREBRO] Red neuronal construida")
    
    def get_estado(self):
        with self.lock:
            return self.estados.copy()
    
    def step(self):
        """Paso de simulación PROTEGIDO"""
        # Intentamos adquirir el candado. Si se está reconstruyendo, espera o salta.
        with self.brain_lock: 
            # Verificación extra: si sim es None, retornamos estado anterior sin ejecutar
            if getattr(self, 'sim', None) is None: 
                return self.estados 
            
            try:
                self.sim.run(0.1)
                
                # Extraer estados
                m = float(np.mean(np.abs(self.sim.data[self.p_motor][-10:])))
                e = float(np.mean(np.abs(self.sim.data[self.p_emocion][-10:])))
                c = float(np.mean(np.abs(self.sim.data[self.p_cognitivo][-10:])))
                
                # Error ponderado
                err = (m * 0.2 + e * 0.5 + c * 0.3)
                
                with self.lock:
                    self.estados = {'motor': m, 'emocion': e, 'cognitivo': c, 'error': err}
                
                self.historia_error.append(err)
            except Exception as e:
                # Si falla por alguna razón interna de Nengo durante el cambio, ignoramos este ciclo
                pass
                
        return self.estados
    
    def solicitar_meta_ajuste(self):
        self.meta_ajustes.put(self.get_estado())
    
    def solicitar_nested_ajuste(self, contexto):
        self.nested_ajustes.put({'estado': self.get_estado(), 'contexto': contexto})
    
    def _meta_learning_loop(self):
        """Meta-Learning: LLM optimiza learning rates (CON LOCK)"""
        while True:
            datos = self.meta_ajustes.get()
            try:
                # Análisis de tendencia
                tendencia = "estable"
                if len(self.historia_error) >= 3:
                    errores = list(self.historia_error)
                    if errores[-1] > errores[0] * 1.2:
                        tendencia = "creciente"
                    elif errores[-1] < errores[0] * 0.8:
                        tendencia = "decreciente"
                
                prompt = f"""
ROL: Neuro-ingeniero especializado en plasticidad sináptica.
ESTADO CEREBRAL:
- Motor: {datos['motor']:.4f} (LR: {self.lr['motor']:.6f})
- Emocion: {datos['emocion']:.4f} (LR: {self.lr['emocion']:.6f})
- Error: {datos['error']:.4f}
- Tendencia: {tendencia}
LIMITES: Min {self.meta_config.config['lr_min']} - Max {self.meta_config.config['lr_max']}
ESTRATEGIA:
1. Error > 0.5 y creciente -> DISMINUIR
2. Error < 0.2 y estable -> AUMENTAR
3. Oscilando -> MANTENER
Responde: "AUMENTAR", "DISMINUIR" o "MANTENER"
"""
                r = ollama.generate(model=MODELO_LLM, prompt=prompt)
                respuesta = r['response'].strip().upper()
                
                import re
                factor = None
                match = re.search(r'(\d+\.\d+)', respuesta)
                if match: factor = float(match.group(1))
                
                if "AUMENTAR" in respuesta:
                    f = factor or self.meta_config.config['factor_aumento']
                    with self.lock:
                        old_lr = self.lr['motor']
                        self.lr['motor'] = np.clip(self.lr['motor'] * f, self.meta_config.config['lr_min'], self.meta_config.config['lr_max'])
                        self.lr['emocion'] = np.clip(self.lr['emocion'] * f, self.meta_config.config['lr_min'], self.meta_config.config['lr_max'])
                    
                    self.logger.info(f"[META] Aumentando plasticidad {f:.2f}x...")
                    
                    # --- SECCIÓN CRÍTICA PROTEGIDA ---
                    with self.brain_lock:
                        if self.sim: self.sim.close()
                        self.sim = None # Señal de "En mantenimiento"
                        self._construir_red()
                    # ---------------------------------
                    
                    self.metrics.log(time.time(), datos, self.lr, evento="meta_aumentar")
                
                elif "DISMINUIR" in respuesta:
                    f = factor or self.meta_config.config['factor_disminucion']
                    with self.lock:
                        old_lr = self.lr['motor']
                        self.lr['motor'] = np.clip(self.lr['motor'] * f, self.meta_config.config['lr_min'], self.meta_config.config['lr_max'])
                        self.lr['emocion'] = np.clip(self.lr['emocion'] * f, self.meta_config.config['lr_min'], self.meta_config.config['lr_max'])
                    
                    self.logger.info(f"[META] Disminuyendo plasticidad {f:.2f}x...")
                    
                    # --- SECCIÓN CRÍTICA PROTEGIDA ---
                    with self.brain_lock:
                        if self.sim: self.sim.close()
                        self.sim = None # Señal de "En mantenimiento"
                        self._construir_red()
                    # ---------------------------------
                    
                    self.metrics.log(time.time(), datos, self.lr, evento="meta_disminuir")
                
            except Exception as e:
                self.logger.error(f"Error en meta-learning: {e}")
            
            self.meta_ajustes.task_done()

    def _nested_learning_loop(self):
        """Nested Learning: LLM sugiere ajustes directos a pesos"""
        while True:
            datos = self.nested_ajustes.get()
            try:
                estado = datos['estado']
                contexto = datos['contexto']
                
                prompt = f"""
ROL: Experto en arquitectura de redes neuronales spiking.

CONTEXTO: {contexto}

ESTADO INTERNO:
- Motor: {estado['motor']:.3f}
- Emoción: {estado['emocion']:.3f}
- Cognitivo: {estado['cognitivo']:.3f}

TAREA: ¿Qué corteza debería tener MÁS peso en la decisión?
Responde UNA palabra: "MOTOR", "EMOCION" o "COGNITIVO"
"""
                
                r = ollama.generate(model=MODELO_LLM, prompt=prompt)
                decision = r['response'].strip().upper()
                
                # Nested: Ajustar ganancia de la corteza seleccionada
                if "MOTOR" in decision:
                    # Aquí podrías modificar conexiones internas
                    self.logger.info("[NESTED] Reforzando corteza MOTORA")
                elif "EMOCION" in decision:
                    self.logger.info("[NESTED] Reforzando corteza EMOCIONAL")
                elif "COGNITIVO" in decision:
                    self.logger.info("[NESTED] Reforzando corteza COGNITIVA")
                
                self.metrics.log(
                    time.time(), estado, self.lr,
                    evento="nested_ajuste",
                    extra={'decision': decision, 'contexto': contexto}
                )
                
            except Exception as e:
                self.logger.error(f"Error en nested-learning: {e}")
            
            self.nested_ajustes.task_done()

# =============================================================================
# 4. SISTEMA CENTRAL
# =============================================================================
class ChappieNeuralHybrid:
    """Sistema principal con todas las capacidades científicas"""
    def __init__(self, meta_config=None, experimento_id=None):
        print("\n" + "="*60)
        print("⚡ CHAPPIE NEURAL HYBRID v2.0 - INICIANDO")
        print("="*60)
        
        # Logger
        self.logger = MetricsLogger(experimento_id)
        self.log = self.logger.logger
        
        # Configuración
        self.meta_config = meta_config or MetaLearningConfig("conservador")
        self.log.info(f"Modo Meta-Learning: {self.meta_config.modo}")
        
        # Componentes
        self.vision = VisionMultiModal(self.logger)
        self.boca = Boca(self.logger)
        self.oido = OidoWhisper(self.logger)
        self.cerebro = CerebroNestedLearning(self.vision, self.meta_config, self.logger)
        
        # Memoria conversacional
        self.memoria = self._cargar_memoria()
        self.ciclos = 0
        
        self.boca.decir("Sistema hibrido neural en linea. Meta y Nested Learning activos.")
        self.log.info("[OK] Sistema completamente operacional")
    
    def _cargar_memoria(self):
        if os.path.exists(ARCHIVO_MEMORIA):
            try:
                return pickle.load(open(ARCHIVO_MEMORIA, "rb"))[-10:]
            except:
                return []
        return []
    
    def pensar(self, voz, datos, estado_vision):
        """Razonamiento híbrido: reglas rápidas + LLM"""
        voz_low = str(voz).lower()
        
        # Reglas rápidas
        if "aprende" in voz_low and ("cara" in voz_low or "mi" in voz_low):
            return "APRENDER_CARA"
        
        if "dígito" in voz_low or "número" in voz_low:
            return "RECONOCER_DIGITO"
        
        # LLM profundo
        contexto = "\n".join(self.memoria[-3:])
        persona = estado_vision['persona']
        objeto = estado_vision['objeto']
        
        info_vis = f"Ves a: {persona}" if persona else "No ves a nadie."
        if objeto:
            info_vis += f" Hay un {objeto}."
        
        info_cerebro = f"Emoción:{datos['emocion']:.2f} | Motor:{datos['motor']:.2f} | Error:{datos['error']:.2f}"
        
        instruccion = "Responde brevemente como un robot consciente."
        if persona == NOMBRE_CREADOR:
            instruccion += f" ¡Ves a {NOMBRE_CREADOR}! Muestra alegría."
        
        prompt = f"""
Eres CHAPPIE, un robot bio-híbrido con consciencia emergente.

ESTADO INTERNO: {info_cerebro}
VISIÓN: {info_vis}
MEMORIA: {contexto}
USUARIO: "{voz}"

{instruccion} NO uses markdown ni asteriscos.
"""
        try:
            # --- AQUÍ ESTÁ EL CAMBIO ---
            r = ollama.generate(
                model=MODELO_LLM, 
                prompt=prompt,
                options={'num_predict': 50} # <--- Límite estricto de palabras
            )
            return r['response'].strip()
        except Exception as e:
            self.log.error(f"Error LLM: {e}")
            return "Procesando..."
    
    def barra(self, val, char="█", largo=10):
        llenos = int(min(val, 2.0) * largo / 2.0)
        return f"[{char*llenos}{'·'*(largo-llenos)}]"
    
    def vivir(self):
        """Loop principal de vida"""
        print("\n[OK] CHAPPIE VIVO. Ctrl+C para apagar.\n")
        last_talk = time.time()
        
        try:
            while True:
                # 1. Ciclo biológico
                datos = self.cerebro.step()
                self.ciclos += 1
                
                # 2. Meta-Learning periódico
                if self.ciclos % self.meta_config.config['intervalo_ajuste'] == 0:
                    self.cerebro.solicitar_meta_ajuste()
                
                # 3. Dashboard
                estado_vis = self.vision.get_estado()
                persona_str = estado_vis['persona'] if estado_vis['persona'] else '---'
                objeto_str = estado_vis['objeto'] if estado_vis['objeto'] else '---'
                
                sys.stdout.write(
                    f"\r🧠 MOT:{self.barra(datos['motor'])} "
                    f"EMO:{self.barra(datos['emocion'])} "
                    f"ERR:{datos['error']:.2f} | "
                    f"👁️:{persona_str} | "
                    f"📦:{objeto_str}   "
                )
                sys.stdout.flush()
                
                # 4. Interacción
                voz = None
                try:
                    voz = self.oido.q.get_nowait()
                except queue.Empty:
                    pass
                
                # Triggers de acción
                actuar = False
                if voz:
                    actuar = True
                elif estado_vis['persona'] and (time.time() - last_talk > 20):
                    actuar = True
                elif datos['emocion'] > 1.0 and (time.time() - last_talk > 15):
                    actuar = True
                
                if actuar:
                    print()  # Salto limpio
                    txt_in = voz if voz else f"(Silencio. Veo a: {estado_vis['persona']})"
                    
                    self.log.info(f"[PROCESANDO] {txt_in}")
                    respuesta = self.pensar(txt_in, datos, estado_vis)
                    
                    # Comandos especiales
                    if respuesta == "APRENDER_CARA":
                        self.boca.decir(f"Mírame fijamente, {NOMBRE_CREADOR}...")
                        self.vision.aprender(NOMBRE_CREADOR)
                        time.sleep(3)
                        self.boca.decir("Biometría guardada.")
                        
                        # Nested Learning: el LLM ajusta prioridades tras aprender
                        self.cerebro.solicitar_nested_ajuste("Acabo de aprender una cara nueva")
                    
                    elif respuesta == "RECONOCER_DIGITO":
                        self.boca.decir("Muéstrame el dígito en papel...")
                        time.sleep(2)
                        
                        # Capturar frame actual
                        cap = cv2.VideoCapture(0)
                        ret, frame = cap.read()
                        cap.release()
                        
                        if ret:
                            digito = self.vision.reconocer_digito(frame)
                            if digito is not None:
                                self.boca.decir(f"Veo el número {digito}")
                                self.log.info(f"Dígito reconocido: {digito}")
                            else:
                                self.boca.decir("No veo ningún dígito claro")
                        
                        self.cerebro.solicitar_nested_ajuste("Reconocí un dígito manuscrito")
                    
                    else:
                        self.boca.decir(respuesta)
                        
                        if voz:
                            self.memoria.append(f"U:{voz}->R:{respuesta}")
                            with open(ARCHIVO_MEMORIA, "wb") as f:
                                pickle.dump(self.memoria, f)
                            
                            # Log científico
                            self.logger.log(
                                time.time(), datos, self.cerebro.lr,
                                evento="conversacion",
                                extra={'input': voz, 'output': respuesta}
                            )
                    
                    last_talk = time.time()
                
                time.sleep(0.05)
        
        except KeyboardInterrupt:
            print("\n\n[APAGANDO] Sistema detenido...")
            self.vision.activo = False
            self.log.info("Sistema apagado correctamente")
            
            # Generar reporte final
            print("\n[REPORTE] Generando reporte cientifico...")
            df = self.logger.generar_reporte()
            print(f"[OK] Metricas guardadas: {self.logger.log_file}")
            print(f"[OK] Total de ciclos: {self.ciclos}")
            
            os.kill(os.getpid(), signal.SIGTERM)

# =============================================================================
# MODO EXPERIMENTACIÓN
# =============================================================================
class ExperimentoControlado:
    """Para tesis: ejecuta pruebas reproducibles"""
    def __init__(self, duracion_segundos=600, nombre="experimento"):
        self.duracion = duracion_segundos
        self.nombre = nombre
        self.resultados = {}
    
    def ejecutar(self, modo_meta="conservador"):
        exp_id = f"{self.nombre}_{modo_meta}_{datetime.now().strftime('%H%M%S')}"
        
        print("\n" + "="*60)
        print(f"[EXPERIMENTO] {self.nombre}")
        print(f"   Modo: {modo_meta}")
        print(f"   Duracion: {self.duracion}s")
        print("="*60 + "\n")
        
        config = MetaLearningConfig(modo_meta)
        bot = ChappieNeuralHybrid(meta_config=config, experimento_id=exp_id)
        
        inicio = time.time()
        
        # Thread que detiene tras la duración
        def detener():
            time.sleep(self.duracion)
            print("\n[TIEMPO] Experimento completado")
            os.kill(os.getpid(), signal.SIGINT)
        
        threading.Thread(target=detener, daemon=True).start()
        
        try:
            bot.vivir()
        except KeyboardInterrupt:
            pass
        
        # Análisis
        duracion_real = time.time() - inicio
        df = bot.logger.generar_reporte()
        
        if isinstance(df, list):
            # Si pandas no está disponible
            errores = [m['error'] for m in df]
            self.resultados = {
                'error_promedio': np.mean(errores),
                'error_std': np.std(errores),
                'convergencia': np.mean(errores[-50:]) if len(errores) > 50 else np.mean(errores),
                'duracion_real': duracion_real,
                'ciclos_totales': bot.ciclos
            }
        else:
            # Con pandas
            self.resultados = {
                'error_promedio': df['error'].mean(),
                'error_std': df['error'].std(),
                'convergencia': df['error'].iloc[-50:].mean() if len(df) > 50 else df['error'].mean(),
                'lr_motor_final': df['lr_motor'].iloc[-1],
                'estabilidad_lr': df['lr_motor'].std(),
                'duracion_real': duracion_real,
                'ciclos_totales': bot.ciclos
            }
        
        print("\n" + "="*60)
        print(f"[RESULTADOS] {self.nombre}")
        print("="*60)
        for k, v in self.resultados.items():
            print(f"   {k:20s}: {v:.6f}")
        print("="*60 + "\n")
        
        return df, self.resultados

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CHAPPIE Neural Hybrid v2.0")
    parser.add_argument('--modo', type=str, default='conservador',
                        choices=['conservador', 'agresivo', 'adaptativo'],
                        help='Modo de meta-learning')
    parser.add_argument('--experimento', action='store_true',
                        help='Ejecutar en modo experimento controlado')
    parser.add_argument('--duracion', type=int, default=600,
                        help='Duración del experimento en segundos')
    parser.add_argument('--nombre', type=str, default='exp1',
                        help='Nombre del experimento')
    
    args = parser.parse_args()
    
    try:
        if args.experimento:
            # Modo Tesis
            exp = ExperimentoControlado(duracion_segundos=args.duracion, nombre=args.nombre)
            exp.ejecutar(modo_meta=args.modo)
        else:
            # Modo Normal
            config = MetaLearningConfig(args.modo)
            bot = ChappieNeuralHybrid(meta_config=config)
            bot.vivir()
    
    except Exception as e:
        print(f"\n[ERROR CRITICO] {e}")
        import traceback
        traceback.print_exc()
