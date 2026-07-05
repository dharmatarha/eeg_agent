# EEG-ADK Analysis Studio (Frontend)

This is the Next.js/React frontend application for the EEG-ADK Multi-Agent System. It provides a real-time, dark-themed interactive dashboard allowing researchers to run analysis, review plans, and monitor EEG visualizations.

---

## Key Features

1. **Interactive File Browser**: Browse raw EEG recordings and BIDS directories dynamically inside the mounted `./data` host folder.
2. **Analysis Setup Studio**: Configure subject directives and link previous runs' parameters seamlessly.
3. **Real-time Event Streaming**: Powered by WebSockets to stream agent logs, execution phases, and Matplotlib base64 figures instantly.
4. **Interactive Plan Review Card**: A custom Human-in-the-Loop review component to approve proposed plans or type corrective feedback.
5. **State Hydration**: Queries the FastAPI backend to reload the exact state, chat history, and visual results when revisiting an active or past thread.

---

## Technical Stack

* **Framework**: React / Next.js 16 (App Router)
* **Language**: TypeScript
* **State Management & UI Core**: `assistant-ui` (ExternalStoreRuntime) for chat logic
* **Styling**: TailwindCSS & CSS Modules
* **Build Targets**: Configured in `standalone` output mode in `next.config.ts` for optimized Docker image sizes (~150MB).

---

## Developer Guide

### 1. Requirements
* Node.js v20.9.0 or higher
* npm v10 or higher

### 2. Configure Local API Endpoint
Create a `ui/.env.local` file to specify the address of your local FastAPI backend:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 3. Start Development Server
Install dependencies and boot Next.js in development mode:
```bash
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to view the application.

### 4. Build and Production Run
To build the optimized production package:
```bash
npm run build
npm run start
```

---

## Docker Production Integration

The application is built inside a multi-stage Docker container using [ui/Dockerfile](file:///home/aboncz/workspace/eeg_agent/ui/Dockerfile):
* **Build Arg**: Takes `NEXT_PUBLIC_API_URL` during build-time.
* **Size Optimization**: Only packages the standalone Node build, discarding `devDependencies` and temporary builder layers.
* **To run via Compose**:
  ```bash
  docker compose -f docker/docker-compose.yml up -d frontend
  ```
