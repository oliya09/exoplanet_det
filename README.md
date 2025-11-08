# 🚀 ExoVision — AI Exoplanet Analysis with Real Images and Habitability Assessment

![ExoVision](https://raw.githubusercontent.com/root3315/TIC-id/main/banner.png)

> **Search. Analyze. Visualize.**  
> ExoVision is a modern web application for exoplanet research using data from NASA, SIMBAD, Exoplanet.eu, and artificial intelligence.


## ✨ Features

| Feature | Description |
|--------|--------|
| **🔍 Multi-source Search** | Data from NASA Exoplanet Archive, SIMBAD, Exoplanet.eu |
| **🌱 Habitability Assessment** | 0–100 score + survival chance (%) |
| **🛰️ Real Star Images** | Embedded images from NASA SkyView (2MASS), Hubble, JWST |
| **🎨 Synthetic Images** | Star generation based on temperature and radius |
| **🤖 AI Analysis (Ollama)** | Local LLM connection (gemma2, llama3, etc.) |
| **📊 Visualizations** | Orbit, mass-radius, habitability diagram |
| **💾 Data Download** | JSON + all images |
| **🖥️ Responsive UI** | React + Tailwind + Framer Motion |

## 🛠️ Technologies

### Backend (FastAPI)
```text
Python 3.11+ | FastAPI | Motor (MongoDB) | httpx | PIL | matplotlib
```
### Frontend (React + Vite)
```text
React 18 | TypeScript | Tailwind CSS | Framer Motion | Lucide Icons | Sonner
```

### AI
```text
Ollama (Local LLM) — gemma3
```

## ⚡ Installation
### 1. Clone the repository

```bash
git clone https://github.com/root3315/TIC-id.git
cd TIC-id
```

### 2. Start the backend
```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --reload --port 8001
```

### 3. Start the frontend
```bash
cd ../frontend
npm install
npm run dev
```

### 4. (Optional) Start Ollama

```bash
ollama push gemma3
```

## 📡 API

| Method  | Endpoint                   | Description               |
| ------  | -------------------------- | ------------------------- |
| `POST`  | `/api/search`              | Search by name or TIC ID  |
| `POST`  | `/api/analyze`             | AI analysis via Ollama    |
| `GET`   | `/api/habitability/{name}` | Habitability score only   |


## 📝 Sample Data

```json
{
  "name": "Kepler-186 f",
  "habitability_score": {
    "total_score": 46.0,
    "survival_chance": 39.0,
    "category": "Moderate"
  },
  "visualizations": {
    "synthetic_star": "data:image/png;base64,...",
    "real_images": [
      {
        "source": "NASA SkyView (2MASS)",
        "url": "data:image/png;base64,...",
        "description": "Infrared field around Kepler-186"
      }
    ]
  }
}

```


## 📜 License

[MIT License](LICENSE)


## 👨‍💻 Author
**Stepan Ionichev**

GitHub: [@root3315](https://github.com/root3315)
Telegram: [@Pumpkin2008](https://t.me/@Pumpkin2008)


> **ExoVision** is more than a catalog. It's a tool for future colonizers and space explorers. 🌌
