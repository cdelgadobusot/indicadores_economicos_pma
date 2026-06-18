# ⚡ Guía rápida — Cómo correr todo desde la terminal

Esta guía es el **paso a paso exacto** para ejecutar el proyecto. Copia y pega los
comandos. Todos se ejecutan **desde la carpeta del proyecto** (`parcial2/`).

> Para entrar a la carpeta del proyecto:
> ```bash
> cd /Users/carlosdelgado/Documents/utp/1-semestre-2026/gestioninformacion/parcial2
> ```

---

## 🟢 Forma fácil (con los scripts)

Hay 5 scripts que hacen todo por ti. **Solo necesitas estos comandos:**

```bash
# 1) Instalar todo (crea el entorno e instala dependencias) — solo la 1.ª vez
bash scripts/instalar.sh

# 2) Preparar el chatbot GRATIS con Ollama (solo la 1.ª vez)
bash scripts/preparar_ollama.sh

# 3) Abrir el dashboard interactivo  (Ctrl+C para detener)
bash scripts/dashboard.sh
```

Otros scripts útiles:

```bash
bash scripts/notebook.sh    # abre el notebook en Jupyter
bash scripts/chatbot.sh     # prueba el chatbot en la terminal (sin dashboard)
```

---

## 🔧 Forma manual (paso a paso, sin scripts)

Por si quieres entender qué hace cada paso.

### 1. Instalar (una sola vez)

```bash
python3 -m venv .venv                       # crea el entorno virtual
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

### 2. Chatbot GRATIS con Ollama (una sola vez)

```bash
# (Ollama ya debe estar instalado: https://ollama.com/download)
ollama pull llama3.2                         # descarga el modelo de IA local
```

> Si tu computadora tiene poca memoria (≤ 8 GB RAM), usa un modelo más liviano:
> `ollama pull llama3.2:1b`  y luego  `export OLLAMA_MODELO=llama3.2:1b`

### 3. Ejecutar el dashboard

```bash
./.venv/bin/streamlit run dashboard/app.py   # abre http://localhost:8501
```

En la pestaña **💬 Chatbot RAG** debe aparecer **«🟢 Ollama (IA local, gratis)»**.

### 4. Ejecutar el notebook

```bash
./.venv/bin/jupyter notebook notebooks/proyecto_indicadores_economicos.ipynb
```

### 5. Probar solo el chatbot (en la terminal)

```bash
./.venv/bin/python -m src.rag.chatbot
```

---

## ⚙️ Opciones del chatbot (variables de entorno)

Defínelas **antes** de lanzar el dashboard, en la misma terminal:

```bash
export CHATBOT_PROVEEDOR=ollama     # forzar Ollama (gratis). Otros: auto | claude | extractivo
export OLLAMA_MODELO=qwen2.5:3b     # cambiar de modelo de IA local
export ANTHROPIC_API_KEY=tu-clave   # (opcional) usar Claude en la nube en vez de Ollama
```

Por defecto (`auto`) el chatbot usa: **Ollama** → si no, **Claude** → si no,
**extractivo**.

---

## 🆘 Problemas comunes

| Problema | Solución |
|----------|----------|
| **`AttributeError: ... 'proveedor_activo'`** o cambios que "no se ven" en el dashboard | El servidor quedó con código viejo en caché. **Detén Streamlit** (`Ctrl+C` en la terminal) y vuelve a correr `bash scripts/dashboard.sh`. |
| El chatbot dice **«extractivo»** y querías Ollama | Asegúrate de que la app **Ollama esté abierta/corriendo** y de haber hecho `ollama pull llama3.2`. Verifica con `ollama list`. |
| **La primera respuesta del chatbot tarda** | Normal: Ollama carga el modelo en memoria la primera vez; luego va rápido. |
| **Streamlit pide un correo** la primera vez | Presiona Enter para omitir, o crea `~/.streamlit/credentials.toml` con `[general]\nemail = ""`. |
| **`command not found: ollama`** | Falta instalar Ollama: https://ollama.com/download |
| **El pipeline dice «se usaron datos de respaldo»** | No había internet para la API del Banco Mundial; el proyecto usa datos de respaldo y **funciona igual**. Con internet, baja los datos en vivo. |
| **`python3: command not found`** | Instala Python 3 (https://www.python.org/downloads/) y vuelve a intentar. |

---

## 📋 Resumen ultra-corto

```bash
cd /Users/carlosdelgado/Documents/utp/1-semestre-2026/gestioninformacion/parcial2
bash scripts/instalar.sh          # 1.ª vez
bash scripts/preparar_ollama.sh   # 1.ª vez (chatbot gratis)
bash scripts/dashboard.sh         # cada vez que quieras usarlo
```

¡Eso es todo! 🎉
