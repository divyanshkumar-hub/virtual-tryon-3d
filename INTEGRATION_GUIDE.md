# V-STYLE 3D: FRONTEND & BACKEND INTEGRATION GUIDE
### How to Combine Your AI Model with Your Partner's Frontend

---

## 1. The High-Level Architecture (How They Connect)

```
┌─────────────────────────────────┐                 ┌─────────────────────────────────┐
│        FRONTEND (Partner)       │                 │       BACKEND / AI (You)        │
│  - Three.js 3D Mannequin        │                 │  - Python REST Server           │
│  - Wardrobe UI & Buttons        │                 │  - AI Inpainting & Warping      │
│  - TextureLoader Engine         │                 │  - VITON Dataset & Metrics      │
└────────────────┬────────────────┘                 └────────────────┬────────────────┘
                 │                                                   │
                 │ 1. POST /api/tryon (JSON: garment, body)          │
                 │──────────────────────────────────────────────────►│
                 │                                                   │
                 │                                                   │ 2. Runs AI Pipeline
                 │                                                   │    Generates texture file
                 │                                                   │
                 │ 3. Response: { texture_url: "/samples/..." }      │
                 │◄──────────────────────────────────────────────────│
                 │                                                   │
                 │ 4. THREE.TextureLoader().load(texture_url)        │
                 │    Applies texture to 3D Mannequin mesh           │
                 ▼                                                   ▼
```

---

## 2. The Code Contract (API Specification)

To ensure you and your frontend partner don't have communication bugs, agree on this exact contract:

### A. Frontend Request (What the partner sends):
```javascript
// JavaScript in frontend:
const response = await fetch('/api/tryon', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    garment_id: 'cyber-hoodie',
    color: '#e11d48'
  })
});

const data = await response.json();
console.log("AI Model Texture URL:", data.texture_url);
```

### B. Backend Response (What your Python server returns):
```json
{
  "success": true,
  "message": "AI Try-On synthesis completed",
  "texture_url": "/samples/4_vton_synthesis_result.jpg",
  "metrics": {
    "ssim": 0.892,
    "lpips": 0.079,
    "latency": "1.42s"
  }
}
```

### C. Mapping the Image onto the 3D Model (Three.js):
Your frontend partner applies the texture to the 3D clothing mesh like this:
```javascript
const loader = new THREE.TextureLoader();
loader.load(data.texture_url, function(texture) {
  // Apply image to the 3D shirt/jacket material
  shirtMesh.material.map = texture;
  shirtMesh.material.needsUpdate = true;
});
```

---

## 3. How to Work Together Using GitHub (Step-by-Step)

To collaborate with your 3 partners without overwriting each other's code, follow this standard Git workflow:

### Step 1: Initialize Git and Push to GitHub (You do this once)
Open your terminal in `c:\code\virtual_tryon_3d\`:
```bash
git init
git add .
git commit -m "feat: complete Phase 1 virtual try-on studio and AI pipeline"
git branch -M main
```
Go to [GitHub.com](https://github.com), create a new private or public repository named `virtual-tryon-3d`, and push:
```bash
git remote add origin https://github.com/YOUR_USERNAME/virtual-tryon-3d.git
git push -u origin main
```

### Step 2: Your Partners Clone the Repository
Your frontend partner opens terminal on their laptop and runs:
```bash
git clone https://github.com/YOUR_USERNAME/virtual-tryon-3d.git
cd virtual-tryon-3d
```

### Step 3: Division of Folders (Prevents Merge Conflicts!)
Agree with your team on this division:
* **Frontend Partner:** Modifies only files inside `web_app/` (HTML, CSS, 3D lights, cameras, styles).
* **AI / Backend Partner (You):** Modifies `server.py`, `run_pipeline_local.py`, and `Phase1_AI_Model_Pipeline.ipynb`.
* **Data Partner:** Collects more clothes / images in `samples/`.

Because you are editing different files, Git will **never** have merge conflicts!

### Step 4: Daily Git Routine
Whenever anyone makes a change:
```bash
# 1. Before you start working, pull teammates' latest updates:
git pull

# 2. After you finish a feature, save and push:
git add .
git commit -m "Added new 3D sunglasses accessory"
git push
```

