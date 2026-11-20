import { useState, useRef, useEffect, useMemo, Suspense, Component } from "react";
import * as THREE from "three";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, ContactShadows, useGLTF } from "@react-three/drei";
import "./App.css";

const API_BASE = typeof window !== "undefined" && (window.location.port === "5000" || window.location.port === "") ? "" : "http://127.0.0.1:5000";

/**
 * Procedural 3D Mannequin Fallback
 * Renders if GLB model is loading or in offline mode.
 */
function ProceduralMannequin({ topColor, pantsColor, shoeColor, hasSunglasses, glassesFrameColor }) {
  return (
    <group position={[0, -0.95, 0]}>
      {/* Head */}
      <mesh position={[0, 1.7, 0]}>
        <sphereGeometry args={[0.13, 32, 32]} />
        <meshStandardMaterial color="#d4a373" roughness={0.6} />
      </mesh>
      {/* Neck */}
      <mesh position={[0, 1.52, 0]}>
        <cylinderGeometry args={[0.05, 0.06, 0.1, 16]} />
        <meshStandardMaterial color="#c68b59" roughness={0.6} />
      </mesh>
      {/* Torso / Top */}
      <mesh position={[0, 1.18, 0]}>
        <boxGeometry args={[0.38, 0.58, 0.22]} />
        <meshStandardMaterial color={topColor} roughness={0.4} />
      </mesh>
      {/* Arms */}
      <mesh position={[-0.26, 1.18, 0]}>
        <cylinderGeometry args={[0.045, 0.04, 0.55, 16]} />
        <meshStandardMaterial color={topColor} roughness={0.4} />
      </mesh>
      <mesh position={[0.26, 1.18, 0]}>
        <cylinderGeometry args={[0.045, 0.04, 0.55, 16]} />
        <meshStandardMaterial color={topColor} roughness={0.4} />
      </mesh>
      {/* Hips / Pants Upper */}
      <mesh position={[0, 0.82, 0]}>
        <boxGeometry args={[0.34, 0.18, 0.2]} />
        <meshStandardMaterial color={pantsColor} roughness={0.5} />
      </mesh>
      {/* Left Leg */}
      <mesh position={[-0.1, 0.42, 0]}>
        <cylinderGeometry args={[0.065, 0.05, 0.68, 16]} />
        <meshStandardMaterial color={pantsColor} roughness={0.5} />
      </mesh>
      {/* Right Leg */}
      <mesh position={[0.1, 0.42, 0]}>
        <cylinderGeometry args={[0.065, 0.05, 0.68, 16]} />
        <meshStandardMaterial color={pantsColor} roughness={0.5} />
      </mesh>
      {/* Left Shoe */}
      <mesh position={[-0.1, 0.04, 0.04]}>
        <boxGeometry args={[0.09, 0.08, 0.22]} />
        <meshStandardMaterial color={shoeColor} roughness={0.3} />
      </mesh>
      {/* Right Shoe */}
      <mesh position={[0.1, 0.04, 0.04]}>
        <boxGeometry args={[0.09, 0.08, 0.22]} />
        <meshStandardMaterial color={shoeColor} roughness={0.3} />
      </mesh>

      {/* Sunglasses */}
      {hasSunglasses && (
        <group position={[0, 1.72, 0.12]}>
          <mesh position={[-0.04, 0, 0]}>
            <boxGeometry args={[0.05, 0.03, 0.015]} />
            <meshStandardMaterial color="#090d16" metalness={0.95} roughness={0.1} />
          </mesh>
          <mesh position={[0.04, 0, 0]}>
            <boxGeometry args={[0.05, 0.03, 0.015]} />
            <meshStandardMaterial color="#090d16" metalness={0.95} roughness={0.1} />
          </mesh>
          <mesh position={[0, 0.005, 0]}>
            <boxGeometry args={[0.03, 0.006, 0.01]} />
            <meshStandardMaterial color={glassesFrameColor} metalness={0.9} roughness={0.2} />
          </mesh>
        </group>
      )}
    </group>
  );
}

/**
 * Realistic 3D Human Avatar Component (Ready Player Me GLTF)
 * Traverses model meshes and applies clothing, shoe colors, and 3D sunglasses.
 */
function GlbAvatar({ topColor, pantsColor, shoeColor, hasSunglasses, glassesFrameColor, tryOnActive }) {
  const { scene } = useGLTF("/models/realistic_human.glb");
  const clone = useMemo(() => {
    const cl = scene.clone(true);
    cl.traverse((child) => {
      if (child.isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
        if (child.material) {
          child.material = child.material.clone();
        }
      }
    });
    return cl;
  }, [scene]);

  useEffect(() => {
    clone.traverse((child) => {
      if (child.isMesh && child.material) {
        if (child.name === "Wolf3D_Outfit_Top" && tryOnActive) {
          child.material.color.set(topColor);
          child.material.needsUpdate = true;
        } else if (child.name === "Wolf3D_Outfit_Bottom" && tryOnActive) {
          child.material.color.set(pantsColor);
          child.material.needsUpdate = true;
        } else if (child.name === "Wolf3D_Outfit_Footwear" && tryOnActive) {
          child.material.color.set(shoeColor);
          child.material.needsUpdate = true;
        }
      }
    });
  }, [clone, topColor, pantsColor, shoeColor, tryOnActive]);

  return (
    <group position={[0, -0.95, 0]}>
      <primitive object={clone} />

      {/* Aviator / Wayfarer 3D Sunglasses */}
      {hasSunglasses && (
        <group position={[0, 1.723, 0.108]}>
          <mesh position={[-0.033, 0, 0]}>
            <boxGeometry args={[0.048, 0.028, 0.01]} />
            <meshStandardMaterial color="#090d16" metalness={0.95} roughness={0.1} />
          </mesh>
          <mesh position={[0.033, 0, 0]}>
            <boxGeometry args={[0.048, 0.028, 0.01]} />
            <meshStandardMaterial color="#090d16" metalness={0.95} roughness={0.1} />
          </mesh>
          <mesh position={[0, 0.006, 0]}>
            <boxGeometry args={[0.020, 0.005, 0.008]} />
            <meshStandardMaterial color={glassesFrameColor} metalness={0.9} roughness={0.2} />
          </mesh>
        </group>
      )}
    </group>
  );
}

class AvatarErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

function ThreeDAvatar({ selectedClothes, selectedShoes = [], selectedSunglasses = [], tryOnActive }) {
  const clothesText = selectedClothes.join(" ").toLowerCase();
  const shoesText = [...selectedShoes, ...selectedClothes].join(" ").toLowerCase();
  const glassesText = selectedSunglasses.join(" ").toLowerCase();

  let topColor = "#1e3a8a";
  if (tryOnActive && selectedClothes.length > 0) {
    if (clothesText.includes("blazer") || clothesText.includes("navy")) topColor = "#1e3a8a";
    else if (clothesText.includes("hoodie") || clothesText.includes("cyber")) topColor = "#e11d48";
    else if (clothesText.includes("leather") || clothesText.includes("jacket")) topColor = "#0f172a";
    else if (clothesText.includes("white") || clothesText.includes("oxford") || clothesText.includes("linen")) topColor = "#f8fafc";
    else if (clothesText.includes("tee") || clothesText.includes("t-shirt") || clothesText.includes("athletic")) topColor = "#059669";
    else if (clothesText.includes("sweater") || clothesText.includes("knit")) topColor = "#475569";
    else topColor = "#4f46e5";
  }

  let pantsColor = "#1e293b";
  if (tryOnActive && clothesText.includes("chino")) pantsColor = "#b45309";
  else if (tryOnActive && clothesText.includes("cargo")) pantsColor = "#334155";
  else if (tryOnActive && (clothesText.includes("denim") || clothesText.includes("jeans"))) pantsColor = "#1d4ed8";

  let shoeColor = "#0f172a";
  if (tryOnActive && (shoesText.includes("suede") || shoesText.includes("brown") || shoesText.includes("loafer") || shoesText.includes("chelsea") || shoesText.includes("derby") || shoesText.includes("cognac"))) {
    shoeColor = "#78350f";
  } else if (tryOnActive && (shoesText.includes("white") || shoesText.includes("sneaker") || shoesText.includes("trainer") || shoesText.includes("canvas"))) {
    shoeColor = "#f8fafc";
  } else if (tryOnActive && (shoesText.includes("chunky") || shoesText.includes("runner") || shoesText.includes("streetwear"))) {
    shoeColor = "#dc2626";
  } else if (tryOnActive && selectedShoes.length > 0) {
    shoeColor = "#1e293b";
  }

  const hasSunglasses = tryOnActive && selectedSunglasses.length > 0;
  const glassesFrameColor = glassesText.includes("gold") || glassesText.includes("aviator")
    ? "#eab308"
    : glassesText.includes("tortoise")
    ? "#92400e"
    : "#111827";

  return (
    <AvatarErrorBoundary
      fallback={
        <ProceduralMannequin
          topColor={topColor}
          pantsColor={pantsColor}
          shoeColor={shoeColor}
          hasSunglasses={hasSunglasses}
          glassesFrameColor={glassesFrameColor}
        />
      }
    >
      <Suspense
        fallback={
          <ProceduralMannequin
            topColor={topColor}
            pantsColor={pantsColor}
            shoeColor={shoeColor}
            hasSunglasses={hasSunglasses}
            glassesFrameColor={glassesFrameColor}
          />
        }
      >
        <GlbAvatar
          topColor={topColor}
          pantsColor={pantsColor}
          shoeColor={shoeColor}
          hasSunglasses={hasSunglasses}
          glassesFrameColor={glassesFrameColor}
          tryOnActive={tryOnActive}
        />
      </Suspense>
    </AvatarErrorBoundary>
  );
}

// Preload avatar asset
try {
  useGLTF.preload("/models/realistic_human.glb");
} catch {
  // Preload graceful ignore if network offline
}

function App() {
  const [page, setPage] = useState("home");
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [selectedClothes, setSelectedClothes] = useState([]);
  const [selectedShoes, setSelectedShoes] = useState([]);
  const [selectedSunglasses, setSelectedSunglasses] = useState([]);
  const [userHeight, setUserHeight] = useState(175);
  const [userGender, setUserGender] = useState("unisex");
  const [tryOnActive, setTryOnActive] = useState(false);

  // SQLite Database state
  const [userId, setUserId] = useState(1);
  const [userName, setUserName] = useState("StyleMate Demo User");
  const [userEmail, setUserEmail] = useState("demo@stylemate.ai");
  const [userStatus, setUserStatus] = useState("");
  const [savedOutfits, setSavedOutfits] = useState([]);
  const [saveMessage, setSaveMessage] = useState("");
  const [tryOnStatus, setTryOnStatus] = useState("");

  const tryOnSectionRef = useRef(null);

  const fetchSavedOutfits = async (uid = userId) => {
    try {
      const res = await fetch(`${API_BASE}/api/outfits/${uid}`);
      if (res.ok) {
        const data = await res.json();
        setSavedOutfits(data);
      }
    } catch {
      // Backend offline
    }
  };

  useEffect(() => {
    fetchSavedOutfits(1);
  }, []);

  const handleRegisterUser = async (e) => {
    e.preventDefault();
    if (!userName || !userEmail) return;
    try {
      const res = await fetch(`${API_BASE}/api/users`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: userName, email: userEmail }),
      });
      const data = await res.json();
      if (data.user_id) {
        setUserId(data.user_id);
        setUserStatus(`Active Profile: #${data.user_id} (${userName})`);
        fetchSavedOutfits(data.user_id);
      } else if (data.error) {
        setUserStatus(`Note: ${data.error}`);
      }
    } catch {
      setUserStatus("Offline mode (connect to port 5000 Flask backend)");
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError("");
    setTryOnActive(false);
  };

  const selectClothing = (item) => {
    setSelectedClothes((prev) =>
      prev.includes(item) ? prev.filter((c) => c !== item) : [...prev, item]
    );
  };

  const selectShoe = (item) => {
    setSelectedShoes((prev) =>
      prev.includes(item) ? prev.filter((s) => s !== item) : [...prev, item]
    );
  };

  const selectSunglasses = (item) => {
    setSelectedSunglasses((prev) =>
      prev.includes(item) ? prev.filter((g) => g !== item) : [...prev, item]
    );
  };

  const handleTrySelectedOutfit = async () => {
    setPage("tryon");
    setTryOnActive(true);
    try {
      const res = await fetch(`${API_BASE}/api/try-on`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clothing_name: selectedClothes[0] || "Navy Tailored Double-Breasted Blazer",
          accessory_id: selectedSunglasses.length > 0 ? 1 : null,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setTryOnStatus(`${data.message}`);
      }
    } catch {
      // Offline fallback
    }
    if (tryOnSectionRef.current) {
      tryOnSectionRef.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  const handleSaveOutfitToDB = async () => {
    const itemsToSave = [...selectedClothes, ...selectedShoes];
    if (itemsToSave.length === 0) return;
    setSaveMessage("Saving to StyleMate SQLite DB...");
    try {
      for (const itemName of itemsToSave) {
        await fetch(`${API_BASE}/api/outfits/save`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: userId,
            clothing_name: itemName,
            accessory_id: selectedSunglasses.length > 0 ? 1 : null,
          }),
        });
      }
      setSaveMessage("✓ Outfit saved to SQLite database (stylemate.db)!");
      fetchSavedOutfits(userId);
    } catch {
      setSaveMessage("Could not connect to port 5000 backend.");
    }
  };

  const analyzeImage = async () => {
    if (!image) {
      setError("Please upload a full-body image.");
      return;
    }

    setLoading(true);
    setError("");

    const formData = new FormData();
    formData.append("image", image);
    formData.append("user_id", String(userId));
    formData.append("height_cm", String(userHeight));
    formData.append("gender", userGender);

    try {
      // Connect to either /api/analyze or /api/upload
      let response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        response = await fetch(`${API_BASE}/api/upload`, {
          method: "POST",
          body: formData,
        });
      }

      if (!response.ok) {
        throw new Error("Server error");
      }

      const data = await response.json();
      setResult(data);
    } catch {
      // Graceful local AI analysis fallback
      setResult({
        analysis_id: 1,
        body_type: "Inverted Triangle (V-Shape Athletic)",
        face_shape: "Oval / Defined",
        body_measurements: {
          shoulder_cm: 47.0,
          chest_cm: 100.0,
          waist_cm: 82.0,
          hip_cm: 96.0,
        },
        recommended_clothes: [
          "Navy Tailored Double-Breasted Blazer (Size M)",
          "Cyber Urban Hoodie (Size M)",
          "Slim-Fit White Oxford Shirt (Size M)",
          "Relaxed Straight-Leg Denim Jeans (Size M)",
        ],
        recommended_shoes: [
          "Classic White Leather Sneakers",
          "Suede Chelsea Boots",
          "Chunky Streetwear Runners",
        ],
        recommended_sunglasses: [
          "Gold Aviator Sunglasses",
          "Tortoiseshell Wayfarer Frames",
          "Vintage Clubmaster Shades",
        ],
      });
      setError("Note: Displaying simulated analysis (Run python mini.py on port 5000 for live MediaPipe & SQLite).");
    }

    setLoading(false);
  };

  return (
    <div className="app">
      <nav className="navbar">
        <h1>AI Fashion Stylist &amp; 3D Try-On</h1>

        <div className="nav-links">
          <a href="#home" onClick={() => setPage("home")}>Home</a>
          <a href="#upload" onClick={() => setPage("upload")}>Try Now</a>
          <a href="#tryon" onClick={() => { setPage("tryon"); tryOnSectionRef.current?.scrollIntoView({ behavior: "smooth" }); }}>3D Try-On</a>
          <a href="#about" onClick={() => setPage("about")}>About</a>
        </div>
      </nav>

      <section id="home" className="hero">
        <div>
          <h2>Find Your Perfect Style</h2>
          <p>
            Upload your full-body image for computer vision body measurements &amp;
            face shape analysis, personalized outfit recommendations, and live 3D try-on.
          </p>
          <a href="#upload">
            <button onClick={() => setPage("upload")}>Get Started</button>
          </a>
        </div>
      </section>

      <section id="upload" className="upload-section">
        <h2>Upload Your Full-Body Image</h2>
        <p>AI Computer Vision Landmark Detection &amp; SQLite Profile</p>

        {/* User Profile Registration Card */}
        <div className="upload-box" style={{ marginBottom: "24px", padding: "20px" }}>
          <h3 style={{ marginBottom: "12px" }}>👤 User Profile (#{userId})</h3>
          <form
            onSubmit={handleRegisterUser}
            style={{ display: "flex", gap: "10px", flexWrap: "wrap", justifyContent: "center" }}
          >
            <input
              type="text"
              placeholder="Name"
              value={userName}
              onChange={(e) => setUserName(e.target.value)}
              style={{ padding: "10px", borderRadius: "6px", border: "1px solid #ccc", marginBottom: 0 }}
            />
            <input
              type="email"
              placeholder="Email"
              value={userEmail}
              onChange={(e) => setUserEmail(e.target.value)}
              style={{ padding: "10px", borderRadius: "6px", border: "1px solid #ccc", marginBottom: 0 }}
            />
            <button type="submit" style={{ padding: "10px 18px" }}>
              Switch User
            </button>
          </form>
          {userStatus && (
            <p style={{ marginTop: "10px", color: "#059669", fontWeight: 600 }}>{userStatus}</p>
          )}
        </div>

        <div className="upload-box">
          <div style={{ marginBottom: "20px", padding: "14px", background: "#f8fafc", borderRadius: "8px", border: "1px solid #e2e8f0" }}>
            <h4 style={{ marginBottom: "12px", color: "#334155", fontSize: "15px" }}>📏 Body Calibration (Accurate Fit & Measurements)</h4>
            <div style={{ display: "flex", gap: "16px", justifyContent: "center", flexWrap: "wrap", alignItems: "center" }}>
              <label style={{ display: "flex", flexDirection: "column", gap: "4px", textAlign: "left", fontSize: "13px", fontWeight: "600", color: "#475569" }}>
                Height (cm):
                <input
                  type="number"
                  min="130"
                  max="220"
                  value={userHeight}
                  onChange={(e) => setUserHeight(Number(e.target.value) || 175)}
                  style={{ padding: "8px 12px", borderRadius: "6px", border: "1px solid #cbd5e1", width: "120px", marginBottom: 0 }}
                />
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: "4px", textAlign: "left", fontSize: "13px", fontWeight: "600", color: "#475569" }}>
                Silhouette / Fit:
                <select
                  value={userGender}
                  onChange={(e) => setUserGender(e.target.value)}
                  style={{ padding: "8px 12px", borderRadius: "6px", border: "1px solid #cbd5e1", minWidth: "140px", marginBottom: 0 }}
                >
                  <option value="unisex">Standard / Unisex</option>
                  <option value="male">Men's Sizing</option>
                  <option value="female">Women's Sizing</option>
                </select>
              </label>
            </div>
          </div>

          <input type="file" accept="image/*" onChange={handleImageChange} />

          {preview && (
            <div className="preview">
              <img src={preview} alt="Uploaded preview" />
            </div>
          )}

          <button onClick={analyzeImage} disabled={loading}>
            {loading ? "Analyzing Body & Landmarks..." : "Analyze Image"}
          </button>

          {error && <p className="error">{error}</p>}
        </div>
      </section>

      {result && (
        <section className="results">
          <h2>
            Your AI Analysis {result.analysis_id ? `(Record #${result.analysis_id})` : ""}
          </h2>

          <div className="result-card">
            <h3>Body Information &amp; Measurements</h3>
            <p><strong>Body Type:</strong> {result.body_type}</p>
            {result.body_measurements && (
              <p style={{ fontSize: "14px", color: "#555" }}>
                <strong>Shoulders:</strong> {result.body_measurements.shoulder_cm || 47} cm |{" "}
                <strong>Chest:</strong> {result.body_measurements.chest_cm || 100} cm |{" "}
                <strong>Waist:</strong> {result.body_measurements.waist_cm || 82} cm |{" "}
                <strong>Hips:</strong> {result.body_measurements.hip_cm || 96} cm
              </p>
            )}
            {result.clothing_size && (
              <p><strong>Estimated Size:</strong> {result.clothing_size}</p>
            )}
            {result.body_measurements && (
              <p style={{ fontSize: "12px", color: "#059669", fontWeight: "600", marginTop: "8px" }}>
                ✓ Calibrated for {result.body_measurements.height_cm || userHeight} cm height ({result.body_measurements.gender || userGender})
              </p>
            )}
          </div>

          <div className="result-card">
            <h3>Face &amp; Tone Analysis</h3>
            <p><strong>Face Shape:</strong> {result.face_shape}</p>
            {result.skin_tone && (
              <p><strong>Skin Tone:</strong> {result.skin_tone} ({result.undertone} Undertone)</p>
            )}
          </div>

          <div className="result-card">
            <h3>Recommended Clothes</h3>
            <div className="recommendation-list">
              {result.recommended_clothes?.map((item, index) => (
                <div
                  className={selectedClothes.includes(item) ? "recommendation selected" : "recommendation"}
                  key={index}
                >
                  <span>{item}</span>
                  <button onClick={() => selectClothing(item)}>
                    {selectedClothes.includes(item) ? "Selected ✓" : "Select"}
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="result-card">
            <h3>👟 Recommended Shoes</h3>
            <div className="recommendation-list">
              {result.recommended_shoes?.map((item, index) => (
                <div
                  className={selectedShoes.includes(item) ? "recommendation selected" : "recommendation"}
                  key={index}
                >
                  <span>{item}</span>
                  <button onClick={() => selectShoe(item)}>
                    {selectedShoes.includes(item) ? "Selected ✓" : "Select"}
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="result-card">
            <h3>Recommended Sunglasses</h3>
            <div className="recommendation-list">
              {result.recommended_sunglasses?.map((item, index) => (
                <div
                  className={selectedSunglasses.includes(item) ? "recommendation selected" : "recommendation"}
                  key={index}
                >
                  <span>{item}</span>
                  <button onClick={() => selectSunglasses(item)}>
                    {selectedSunglasses.includes(item) ? "Selected ✓" : "Select"}
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="selected-outfit">
            <h3>Selected Outfit</h3>
            <p>
              <strong>Clothes:</strong>{" "}
              {selectedClothes.length > 0 ? selectedClothes.join(", ") : "None selected"}
            </p>
            <p>
              <strong>Shoes:</strong>{" "}
              {selectedShoes.length > 0 ? selectedShoes.join(", ") : "None selected"}
            </p>
            <p>
              <strong>Sunglasses:</strong>{" "}
              {selectedSunglasses.length > 0 ? selectedSunglasses.join(", ") : "None selected"}
            </p>

            <div style={{ display: "flex", gap: "12px", justifyContent: "center", flexWrap: "wrap", marginTop: "14px" }}>
              <button
                onClick={handleTrySelectedOutfit}
                disabled={
                  selectedClothes.length === 0 &&
                  selectedShoes.length === 0 &&
                  selectedSunglasses.length === 0
                }
              >
                Try Selected Outfit on 3D Model
              </button>

              <button
                onClick={handleSaveOutfitToDB}
                disabled={selectedClothes.length === 0 && selectedShoes.length === 0}
                style={{ background: "#059669" }}
              >
                💾 Save Outfit to SQLite DB
              </button>
            </div>

            {saveMessage && (
              <p style={{ marginTop: "10px", color: "#059669", fontWeight: 600 }}>{saveMessage}</p>
            )}
          </div>
        </section>
      )}

      {/* Saved Outfits */}
      {savedOutfits.length > 0 && (
        <section style={{ padding: "30px 10%", background: "#eae3dc" }}>
          <div className="result-card" style={{ maxWidth: "850px", margin: "auto" }}>
            <h3>🗄️ Saved Outfits in SQLite Database (User #{userId})</h3>
            <div className="recommendation-list" style={{ marginTop: "12px" }}>
              {savedOutfits.map((o) => (
                <div className="recommendation" key={o.outfit_id}>
                  <span>
                    <strong>Outfit #{o.outfit_id}:</strong> {o.clothing_name}{" "}
                    {o.accessory_id ? "(+ Sunglasses)" : ""}
                  </span>
                  <button
                    onClick={() => {
                      setSelectedClothes([o.clothing_name]);
                      if (o.accessory_id) {
                        setSelectedSunglasses(["Gold Aviator Sunglasses"]);
                      }
                      setPage("tryon");
                      setTryOnActive(true);
                      tryOnSectionRef.current?.scrollIntoView({ behavior: "smooth" });
                    }}
                  >
                    Load on 3D Model
                  </button>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* 3D Virtual Try-On Section */}
      <section id="tryon" className="try-on" ref={tryOnSectionRef}>
        <h2>3D Virtual Try-On Studio</h2>

        <div className="avatar-box">
          <div
            style={{
              width: "100%",
              height: "420px",
              background: "radial-gradient(circle, #1e293b 0%, #0f172a 100%)",
              borderRadius: "12px",
              marginBottom: "20px",
              overflow: "hidden",
            }}
          >
            <Canvas shadows camera={{ position: [0, 0.4, 2.8], fov: 45 }}>
              <ambientLight intensity={0.7} />
              <directionalLight position={[3, 4, 4]} intensity={1.2} castShadow />
              <directionalLight position={[-3, 2, -2]} intensity={0.6} color="#818cf8" />
              <ThreeDAvatar
                selectedClothes={selectedClothes}
                selectedShoes={selectedShoes}
                selectedSunglasses={selectedSunglasses}
                tryOnActive={tryOnActive}
              />
              <ContactShadows position={[0, -1.05, 0]} opacity={0.5} scale={5} blur={2} />
              <OrbitControls enablePan={false} minDistance={1.2} maxDistance={4.5} />
            </Canvas>
          </div>

          <p>
            {tryOnActive
              ? `Wearing: ${[...selectedClothes, ...selectedShoes, ...selectedSunglasses].join(", ") || "Default Outfit"} (Drag to rotate 360°)`
              : "Select clothes, shoes, or sunglasses above and click 'Try Selected Outfit' to dress the 3D avatar. Drag with mouse to rotate 360°."}
          </p>

          {tryOnStatus && (
            <p style={{ fontSize: "13px", color: "#4f46e5", marginBottom: "12px" }}>
              API: {tryOnStatus}
            </p>
          )}

          <button onClick={() => { setPage("tryon"); setTryOnActive(true); }}>
            Try Selected Outfit
          </button>
        </div>
      </section>

      <section id="about" className="about">
        <h2>About the Project</h2>
        <p>
          AI Fashion Stylist combines MediaPipe computer vision body measurements,
          Flask + SQLite database persistence, and interactive Three.js 3D virtual try-on.
        </p>
      </section>

      <footer>
        <p>AI Fashion Stylist | 3D Virtual Try-On Studio</p>
      </footer>
    </div>
  );
}

export default App;
